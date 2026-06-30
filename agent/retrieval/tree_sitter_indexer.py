import tree_sitter

from typing import List
from agent.models.schemas import Symbol
from agent.config import logger


class TreeSitterIndexer:
    """Tree-sitter backed symbol extractor.

    Each language grammar is loaded in its own guarded block so a missing or
    unloadable optional grammar (e.g. Go/Rust/Java not installed) only skips
    *that* language with a clear log line, mirroring the Phase 4 tooling_check
    spirit — it never crashes indexing.
    """

    def __init__(self):
        self.parsers = {}
        # ext -> language key used by the extraction dispatch in parse_file.
        self.lang_by_ext = {}

        self._register_python()
        self._register_js_ts()
        self._register_go()
        self._register_rust()
        self._register_java()

        if not self.parsers:
            logger.error("Tree-sitter: no language grammars could be loaded.")

    def _add(self, lang_key: str, parser, *exts: str) -> None:
        for ext in exts:
            self.parsers[ext] = parser
            self.lang_by_ext[ext] = lang_key

    def _register_python(self) -> None:
        try:
            import tree_sitter_python
            lang = tree_sitter.Language(tree_sitter_python.language())
            self._add('python', tree_sitter.Parser(lang), '.py')
        except Exception as e:
            logger.warning(f"Tree-sitter: skipping Python grammar ({e!r}).")

    def _register_js_ts(self) -> None:
        try:
            import tree_sitter_javascript
            js_lang = tree_sitter.Language(tree_sitter_javascript.language())
            self._add('js', tree_sitter.Parser(js_lang), '.js', '.jsx')
        except Exception as e:
            logger.warning(f"Tree-sitter: skipping JavaScript grammar ({e!r}).")

        try:
            import tree_sitter_typescript
            ts_lang = tree_sitter.Language(tree_sitter_typescript.language_typescript())
            self._add('js', tree_sitter.Parser(ts_lang), '.ts')
            tsx_lang = tree_sitter.Language(tree_sitter_typescript.language_tsx())
            self._add('js', tree_sitter.Parser(tsx_lang), '.tsx')
        except Exception as e:
            logger.warning(f"Tree-sitter: skipping TypeScript grammar ({e!r}).")

    def _register_go(self) -> None:
        try:
            import tree_sitter_go
            lang = tree_sitter.Language(tree_sitter_go.language())
            self._add('go', tree_sitter.Parser(lang), '.go')
        except Exception as e:
            logger.warning(f"Tree-sitter: skipping Go grammar ({e!r}).")

    def _register_rust(self) -> None:
        try:
            import tree_sitter_rust
            lang = tree_sitter.Language(tree_sitter_rust.language())
            self._add('rust', tree_sitter.Parser(lang), '.rs')
        except Exception as e:
            logger.warning(f"Tree-sitter: skipping Rust grammar ({e!r}).")

    def _register_java(self) -> None:
        try:
            import tree_sitter_java
            lang = tree_sitter.Language(tree_sitter_java.language())
            self._add('java', tree_sitter.Parser(lang), '.java')
        except Exception as e:
            logger.warning(f"Tree-sitter: skipping Java grammar ({e!r}).")

    def get_supported_extensions(self) -> List[str]:
        return list(self.parsers.keys())

    def parse_file(self, filepath: str, content: str) -> List[Symbol]:
        ext = filepath[filepath.rfind('.'):] if '.' in filepath else ''
        if ext not in self.parsers:
            return []

        parser = self.parsers[ext]
        lang = self.lang_by_ext[ext]
        tree = parser.parse(bytes(content, "utf8"))

        symbols: List[Symbol] = []

        def emit(name: str, sym_type: str, node) -> None:
            symbols.append(Symbol(
                name=name,
                type=sym_type,
                file=filepath,
                line_start=node.start_point.row + 1,
                line_end=node.end_point.row + 1,
                signature=content[node.start_byte:node.end_byte].split('\n')[0][:100],
            ))

        def named(node):
            n = node.child_by_field_name('name')
            return n.text.decode('utf8') if n else None

        def walk(node):
            self._extract(lang, node, named, emit)
            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return symbols

    @staticmethod
    def _extract(lang, node, named, emit) -> None:
        node_type = node.type

        if lang == 'python':
            if node_type == 'function_definition':
                name = named(node)
                if name:
                    is_method = node.parent and node.parent.type == 'class_definition'
                    emit(name, 'method' if is_method else 'function', node)
            elif node_type == 'class_definition':
                name = named(node)
                if name:
                    emit(name, 'class', node)

        elif lang == 'js':
            if node_type in ('function_declaration', 'method_definition'):
                name = named(node)
                if name:
                    emit(name, 'function' if node_type == 'function_declaration' else 'method', node)
            elif node_type == 'class_declaration':
                name = named(node)
                if name:
                    emit(name, 'class', node)
            elif node_type == 'export_statement':
                emit("export_block", 'export', node)

        elif lang == 'go':
            if node_type == 'function_declaration':
                name = named(node)
                if name:
                    emit(name, 'function', node)
            elif node_type == 'method_declaration':
                name = named(node)
                if name:
                    emit(name, 'method', node)
            elif node_type == 'type_spec':
                name = named(node)
                type_child = node.child_by_field_name('type')
                if name and type_child:
                    kind = type_child.type
                    if kind == 'struct_type':
                        emit(name, 'struct', node)
                    elif kind == 'interface_type':
                        emit(name, 'interface', node)
                    else:
                        emit(name, 'class', node)

        elif lang == 'rust':
            if node_type in ('function_item', 'function_signature_item'):
                name = named(node)
                if name:
                    # A fn under an impl/trait declaration_list is a method.
                    parent = node.parent
                    grand = parent.parent if parent else None
                    is_method = grand is not None and grand.type in ('impl_item', 'trait_item')
                    emit(name, 'method' if is_method else 'function', node)
            elif node_type == 'struct_item':
                name = named(node)
                if name:
                    emit(name, 'struct', node)
            elif node_type == 'enum_item':
                name = named(node)
                if name:
                    emit(name, 'enum', node)
            elif node_type == 'trait_item':
                name = named(node)
                if name:
                    emit(name, 'trait', node)

        elif lang == 'java':
            if node_type == 'method_declaration':
                name = named(node)
                if name:
                    emit(name, 'method', node)
            elif node_type == 'class_declaration':
                name = named(node)
                if name:
                    emit(name, 'class', node)
            elif node_type == 'interface_declaration':
                name = named(node)
                if name:
                    emit(name, 'interface', node)
            elif node_type == 'enum_declaration':
                name = named(node)
                if name:
                    emit(name, 'enum', node)
