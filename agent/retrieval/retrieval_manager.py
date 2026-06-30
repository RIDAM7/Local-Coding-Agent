import os
from typing import List, Dict
from pathlib import Path

from agent.models.schemas import Plan, RetrievedContext, RetrievalResult, Symbol, NormalizedDiagnostic
from agent.retrieval.ripgrep_search import RipgrepSearch
from agent.retrieval.symbol_index import SymbolIndex
from agent.retrieval.repository_map import RepositoryMap
from agent.config import settings, logger

MAX_CONTEXT_FILES = 10
MAX_CONTEXT_CHARS = 30000

class RetrievalManager:
    def __init__(self, rg: RipgrepSearch, sym_idx: SymbolIndex, repo_map: RepositoryMap, workspace_path: Path = None):
        self.rg = rg
        self.sym_idx = sym_idx
        self.repo_map = repo_map
        self.workspace = workspace_path if workspace_path else settings.get_workspace_path()
        self.index_dir = os.path.join(self.workspace, "index")

    async def search_context(self, task: str, plan: Plan | None, diagnostic: NormalizedDiagnostic = None) -> RetrievedContext:
        logger.info("Retrieving relevant context files...")
        
        symbols = self.sym_idx.load(self.index_dir) or []
        repo_map_data = self.repo_map.load(self.index_dir)
        
        file_scores: Dict[str, float] = {}
        file_evidence: Dict[str, List[str]] = {}
        file_symbols: Dict[str, List[Symbol]] = {}
        
        def add_score(filepath: str, points: float, evidence: str):
            # Normalize path for dict
            filepath = filepath.replace('\\', '/')
            if filepath not in file_scores:
                file_scores[filepath] = 0.0
                file_evidence[filepath] = []
                file_symbols[filepath] = []
            file_scores[filepath] += points
            file_evidence[filepath].append(evidence)

        keywords = set(task.split())
        if plan:
            keywords.update(plan.summary.split())
            for step in plan.steps:
                keywords.update(step.description.split())
                
        if diagnostic:
            keywords.update(diagnostic.primary_error_message.split())
            for filepath in diagnostic.suspected_files:
                add_score(filepath, 10.0, "Extracted from diagnostic failure trace")
                
        ignore_words = {"the", "a", "an", "and", "or", "to", "in", "with", "create", "update", "delete", "use"}
        search_terms = {k.strip(".,;:()[]{}'\"").lower() for k in keywords if len(k) > 3} - ignore_words
        
        # 1. Ripgrep
        for term in search_terms:
            try:
                rg_results = await self.rg.exact_search(term, str(self.workspace))
                for res in rg_results:
                    filepath = res.get('path', {}).get('text')
                    if filepath:
                        filepath = str(Path(filepath).as_posix())
                        if "index/" in filepath or "reports/" in filepath or "logs/" in filepath:
                            continue
                        add_score(filepath, 1.0, f"Ripgrep match for '{term}'")
            except Exception as e:
                logger.debug(f"Ripgrep failed for term {term}: {e}")

        # 2. Symbol Search
        for symbol in symbols:
            sym_name_lower = symbol.name.lower()
            for term in search_terms:
                if term in sym_name_lower:
                    add_score(symbol.file, 3.0, f"Symbol match '{symbol.name}'")
                    # Avoid duplicate symbol objects in the list
                    if symbol not in file_symbols[symbol.file]:
                        file_symbols[symbol.file].append(symbol)

        # 3. Path Matches
        if repo_map_data:
            for filepath in repo_map_data.files:
                filepath_lower = filepath.lower()
                for term in search_terms:
                    if term in filepath_lower:
                        add_score(filepath, 5.0, f"Path match for '{term}'")

        results = []
        for filepath, score in file_scores.items():
            results.append(RetrievalResult(
                file=filepath,
                score=score,
                evidence=list(set(file_evidence[filepath])),
                matched_symbols=file_symbols[filepath]
            ))
            
        results.sort(key=lambda x: x.score, reverse=True)
        
        final_results = []
        total_chars = 0
        
        for res in results:
            if len(final_results) >= MAX_CONTEXT_FILES:
                break
                
            full_path = self.workspace / res.file
            if not full_path.exists():
                continue
                
            try:
                content = full_path.read_text(encoding='utf-8', errors='replace')
                chars = len(content)
                if total_chars + chars > MAX_CONTEXT_CHARS:
                    if not final_results:
                        pass # ensure at least 1 file
                    else:
                        continue 
                
                total_chars += chars
                final_results.append(res)
            except Exception as e:
                logger.warning(f"Failed to read {res.file} for context limit check: {e}")
                
        logger.info(f"Retrieved {len(final_results)} files ({total_chars} chars) for context.")
        
        return RetrievedContext(
            results=final_results,
            total_files=len(final_results),
            total_chars=total_chars
        )
