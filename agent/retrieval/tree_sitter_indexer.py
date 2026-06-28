import tree_sitter
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript

from typing import List
from agent.models.schemas import Symbol
from agent.config import logger

class TreeSitterIndexer:
    def __init__(self):
        self.parsers = {}
        
        try:
            py_lang = tree_sitter.Language(tree_sitter_python.language())
            self.parsers['.py'] = tree_sitter.Parser(py_lang)
            
            js_lang = tree_sitter.Language(tree_sitter_javascript.language())
            self.parsers['.js'] = tree_sitter.Parser(js_lang)
            self.parsers['.jsx'] = tree_sitter.Parser(js_lang)
            
            ts_lang = tree_sitter.Language(tree_sitter_typescript.language_typescript())
            self.parsers['.ts'] = tree_sitter.Parser(ts_lang)
            
            tsx_lang = tree_sitter.Language(tree_sitter_typescript.language_tsx())
            self.parsers['.tsx'] = tree_sitter.Parser(tsx_lang)
        except Exception as e:
            logger.error(f"Failed to initialize tree-sitter languages: {e}")

    def get_supported_extensions(self) -> List[str]:
        return list(self.parsers.keys())

    def parse_file(self, filepath: str, content: str) -> List[Symbol]:
        ext = filepath[filepath.rfind('.'):] if '.' in filepath else ''
        if ext not in self.parsers:
            return []

        parser = self.parsers[ext]
        tree = parser.parse(bytes(content, "utf8"))
        
        symbols = []
        
        def walk(node):
            node_type = node.type
            name = None
            sym_type = None
            
            if ext == '.py':
                if node_type == 'function_definition':
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        name = name_node.text.decode('utf8')
                        sym_type = 'method' if node.parent and node.parent.type == 'class_definition' else 'function'
                elif node_type == 'class_definition':
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        name = name_node.text.decode('utf8')
                        sym_type = 'class'
            else: # JS/TS/TSX
                if node_type in ['function_declaration', 'method_definition']:
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        name = name_node.text.decode('utf8')
                        sym_type = 'function' if node_type == 'function_declaration' else 'method'
                elif node_type == 'class_declaration':
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        name = name_node.text.decode('utf8')
                        sym_type = 'class'
                elif node_type == 'export_statement':
                    sym_type = 'export'
                    name = "export_block" 

            if name and sym_type:
                symbols.append(Symbol(
                    name=name,
                    type=sym_type,
                    file=filepath,
                    line_start=node.start_point.row + 1,
                    line_end=node.end_point.row + 1,
                    signature=content[node.start_byte:node.end_byte].split('\n')[0][:100]
                ))
                
            for child in node.children:
                walk(child)
                
        walk(tree.root_node)
        return symbols
