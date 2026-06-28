import os
from pathlib import Path

def replace_in_file(filepath, old, new):
    if not os.path.exists(filepath):
        print(f"Skipping {filepath}, not found.")
        return
    content = Path(filepath).read_text(encoding='utf-8')
    content = content.replace(old, new)
    Path(filepath).write_text(content, encoding='utf-8')
    print(f"Updated {filepath}")

# agent/files/core.py
replace_in_file(
    'agent/files/core.py',
    'def __init__(self):\n        self.workspace = settings.get_workspace_path()',
    'def __init__(self, workspace_path: Path = None):\n        self.workspace = workspace_path if workspace_path else settings.get_workspace_path()'
)

# agent/execution/core.py
replace_in_file(
    'agent/execution/core.py',
    'def __init__(self):\n        self.workspace = settings.get_workspace_path()',
    'def __init__(self, workspace_path: Path = None):\n        self.workspace = workspace_path if workspace_path else settings.get_workspace_path()'
)

# agent/reporting/core.py
replace_in_file(
    'agent/reporting/core.py',
    'def __init__(self):\n        self.reports_dir = Path("reports")',
    'def __init__(self, reports_dir: Path = None):\n        self.reports_dir = Path(reports_dir) if reports_dir else Path("reports")\n        os.makedirs(self.reports_dir, exist_ok=True)'
)

# agent/retrieval/retrieval_manager.py
replace_in_file(
    'agent/retrieval/retrieval_manager.py',
    'def __init__(self, rg: RipgrepSearch, sym_idx: SymbolIndex, repo_map: RepositoryMap):\n        self.rg = rg\n        self.sym_idx = sym_idx\n        self.repo_map = repo_map\n        self.workspace = settings.get_workspace_path()',
    'def __init__(self, rg: RipgrepSearch, sym_idx: SymbolIndex, repo_map: RepositoryMap, workspace_path: Path = None):\n        self.rg = rg\n        self.sym_idx = sym_idx\n        self.repo_map = repo_map\n        self.workspace = workspace_path if workspace_path else settings.get_workspace_path()'
)

# agent/repair/rollback.py
replace_in_file(
    'agent/repair/rollback.py',
    'def __init__(self):\n        self.workspace = settings.get_workspace_path()',
    'def __init__(self, workspace_path: Path = None):\n        self.workspace = workspace_path if workspace_path else settings.get_workspace_path()'
)

# agent/repair/coder.py
replace_in_file(
    'agent/repair/coder.py',
    'def __init__(self, llm_client: OllamaClient):\n        self.llm_client = llm_client\n        self.workspace = settings.get_workspace_path()',
    'def __init__(self, llm_client: OllamaClient, workspace_path: Path = None):\n        self.llm_client = llm_client\n        self.workspace = workspace_path if workspace_path else settings.get_workspace_path()'
)

# agent/validation/patch.py
replace_in_file(
    'agent/validation/patch.py',
    'def __init__(self):\n        self.workspace_path = settings.get_workspace_path()',
    'def __init__(self, workspace_path: Path = None):\n        self.workspace_path = workspace_path if workspace_path else settings.get_workspace_path()'
)

# agent/orchestrator.py
old_orch = """    def __init__(self):
        self.llm_client = OllamaClient()
        self.planner = Planner(self.llm_client)
        self.coder = Coder(self.llm_client)
        self.file_manager = FileManager()
        self.executor = Executor()
        self.reporter = Reporter()
        
        # Retrieval components
        self.rg = RipgrepSearch()
        self.ts_indexer = TreeSitterIndexer()
        self.repo_map = RepositoryMap()
        self.sym_idx = SymbolIndex(self.ts_indexer)
        self.retrieval_manager = RetrievalManager(self.rg, self.sym_idx, self.repo_map)
        
        # Validation components
        self.patch_validator = PatchValidator()
        self.build_validator = BuildValidator(self.executor)
        self.lint_validator = LintValidator(self.executor)
        self.test_validator = TestValidator(self.executor)
        
        # Repair components
        self.rollback_manager = RollbackManager()
        self.repair_coder = RepairCoder(self.llm_client)
        self.repair_manager = RepairManager(self.retrieval_manager, self.repair_coder, self.rollback_manager)
        self.constraint_extractor = ConstraintExtractor(self.llm_client)"""

new_orch = """    def __init__(self, workspace_path: Path = None, reports_dir: Path = None):
        ws_path = workspace_path if workspace_path else settings.get_workspace_path()
        self.llm_client = OllamaClient()
        self.planner = Planner(self.llm_client)
        self.coder = Coder(self.llm_client)
        self.file_manager = FileManager(ws_path)
        self.executor = Executor(ws_path)
        self.reporter = Reporter(reports_dir)
        
        # Retrieval components
        self.rg = RipgrepSearch()
        self.ts_indexer = TreeSitterIndexer()
        self.repo_map = RepositoryMap()
        self.sym_idx = SymbolIndex(self.ts_indexer)
        self.retrieval_manager = RetrievalManager(self.rg, self.sym_idx, self.repo_map, ws_path)
        
        # Validation components
        self.patch_validator = PatchValidator(ws_path)
        self.build_validator = BuildValidator(self.executor)
        self.lint_validator = LintValidator(self.executor)
        self.test_validator = TestValidator(self.executor)
        
        # Repair components
        self.rollback_manager = RollbackManager(ws_path)
        self.repair_coder = RepairCoder(self.llm_client, ws_path)
        self.repair_manager = RepairManager(self.retrieval_manager, self.repair_coder, self.rollback_manager)
        self.constraint_extractor = ConstraintExtractor(self.llm_client)"""
replace_in_file('agent/orchestrator.py', old_orch, new_orch)

# Need to make sure Path is imported in orchestrator if not already
orch_content = Path('agent/orchestrator.py').read_text(encoding='utf-8')
if "from pathlib import Path" not in orch_content:
    replace_in_file('agent/orchestrator.py', 'from datetime import datetime', 'from datetime import datetime\nfrom pathlib import Path')
