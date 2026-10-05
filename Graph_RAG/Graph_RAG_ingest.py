import os
import re
import hashlib
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase

from tree_sitter import Language, Parser

# ---------------------------------------------------------------------------
# Tree-sitter language packages
# ---------------------------------------------------------------------------
# Install with:
#   pip install tree-sitter tree-sitter-c tree-sitter-cpp tree-sitter-java \
#       tree-sitter-python tree-sitter-c-sharp tree-sitter-go tree-sitter-sql \
#       tree-sitter-kotlin tree-sitter-javascript tree-sitter-typescript
#
# The imports are deliberately explicit. A missing package produces a useful
# error at startup instead of silently ingesting a language incorrectly.
try:
    import tree_sitter_c as tsc
    import tree_sitter_cpp as tscpp
    import tree_sitter_java as tsjava
    import tree_sitter_python as tspython
    import tree_sitter_c_sharp as tscsharp
    import tree_sitter_go as tsgo
    import tree_sitter_sql as tssql
    import tree_sitter_kotlin as tskotlin
    import tree_sitter_javascript as tsjs
    import tree_sitter_typescript as tsts
except ImportError as exc:
    raise ImportError(
        "One or more Tree-sitter language packages are missing. "
        "Install all packages listed in the file header."
    ) from exc


# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv(override=True)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_URI or not NEO4J_PASSWORD:
    raise ValueError("Missing NEO4J_URI or NEO4J_PASSWORD in .env")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
)


# ============================================================================
# LANGUAGE REGISTRY
# ============================================================================

def make_parser(language):
    return Parser(Language(language))


PARSERS = {
    ".c": make_parser(tsc.language()),
    ".h": make_parser(tsc.language()),
    ".cc": make_parser(tscpp.language()),
    ".cpp": make_parser(tscpp.language()),
    ".cxx": make_parser(tscpp.language()),
    ".hpp": make_parser(tscpp.language()),
    ".hh": make_parser(tscpp.language()),

    ".java": make_parser(tsjava.language()),
    ".py": make_parser(tspython.language()),

    ".cs": make_parser(tscsharp.language()),
    ".go": make_parser(tsgo.language()),

    ".sql": make_parser(tssql.language()),

    ".kt": make_parser(tskotlin.language()),
    ".kts": make_parser(tskotlin.language()),

    ".js": make_parser(tsjs.language()),
    ".jsx": make_parser(tsjs.language()),
    ".mjs": make_parser(tsjs.language()),
    ".cjs": make_parser(tsjs.language()),

    ".ts": make_parser(tsts.language_typescript()),
    ".tsx": make_parser(tsts.language_tsx()),
}

SUPPORTED_EXTENSIONS = set(PARSERS)

LANGUAGE_BY_EXTENSION = {
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".java": "java",
    ".py": "python",
    ".cs": "csharp",
    ".go": "go",
    ".sql": "sql",
    ".kt": "kotlin", ".kts": "kotlin",
    ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
}

IGNORED_DIRECTORIES = {
    # VCS / build output
    ".git", ".svn", ".hg",
    "node_modules", "dist", "build", "out", "target",
    "coverage", ".next", ".nuxt", ".gradle", ".idea",
    "bin", "obj", "vendor",

    # Python
    "__pycache__", ".venv", "venv", "env", ".pytest_cache",
    ".mypy_cache", ".tox",

    # Common generated/cache directories
    ".cache", ".turbo", ".parcel-cache",
}

MAX_FILE_SIZE_MB = float(os.getenv("GRAPH_RAG_MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE = int(MAX_FILE_SIZE_MB * 1024 * 1024)


# ============================================================================
# BASIC HELPERS
# ============================================================================

def normalize_path(path: str) -> str:
    return Path(path).as_posix()


def stable_id(kind: str, path: str, name: str, location: str = "") -> str:
    raw = f"{kind}|{path}|{name}|{location}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode(
        "utf-8", errors="replace"
    )


def line(node) -> int:
    return node.start_point[0] + 1


def column(node) -> int:
    return node.start_point[1] + 1


def extension_of(filepath: str) -> str:
    return Path(filepath).suffix.lower()


def language_of(filepath: str) -> str:
    return LANGUAGE_BY_EXTENSION.get(extension_of(filepath), "unknown")


def get_parser(filepath: str) -> Parser:
    suffix = extension_of(filepath)
    if suffix not in PARSERS:
        raise ValueError(f"Unsupported source extension: {suffix}")
    return PARSERS[suffix]


def safe_identifier(text: str) -> str:
    return text.strip().strip('"').strip("`").strip("'").strip()


def add_unique(items, value):
    if value and value not in items:
        items.append(value)


# ============================================================================
# GRAPH IR
# ============================================================================

class GraphIR:
    def __init__(self):
        self.nodes = {}
        self.relationships = []

    def add_node(self, label, node_id, **properties):
        self.nodes[node_id] = {
            "label": label,
            "id": node_id,
            **properties,
        }

    def add_relationship(self, source_id, relation, target_id, **properties):
        self.relationships.append({
            "source": source_id,
            "type": relation,
            "target": target_id,
            "properties": properties,
        })


# ============================================================================
# GENERIC TREE-SITTER HELPERS
# ============================================================================

FUNCTION_NODES = {
    "function_declaration",
    "function_definition",
    "function_item",
    "method_declaration",
    "method_definition",
    "constructor_declaration",
    "function",
    "function_signature",
    "method",
    "arrow_function",
    "function_expression",
    "local_function_statement",
}

CLASS_NODES = {
    "class_declaration",
    "class_definition",
    "class_specifier",
    "struct_specifier",
    "struct_declaration",
    "interface_declaration",
    "enum_declaration",
    "enum_specifier",
    "record_declaration",
    "object_declaration",
    "type_declaration",
    "type_spec",
    "trait_declaration",
}

IMPORT_NODES = {
    "import_statement",
    "import_declaration",
    "import_spec",
    "import_clause",
    "using_directive",
    "preproc_include",
    "include",
    "package_import",
    "import_list",
    "import_header",
}

CALL_NODES = {
    "call_expression",
    "call",
    "method_invocation",
    "function_call",
    "invocation_expression",
}

CONTROL_CALL_NODES = {
    "new_expression",
    "object_creation_expression",
    "constructor_invocation",
}

PARAMETER_NODES = {
    "formal_parameters",
    "parameters",
    "parameter_list",
    "parameter_declaration",
    "formal_parameter",
}


def field_or_first_named(node, field_names):
    for name in field_names:
        child = node.child_by_field_name(name)
        if child is not None:
            return child
    return node.named_children[0] if node.named_children else None


def node_name(node, source: bytes) -> Optional[str]:
    """
    Extract a useful symbol name across Tree-sitter grammars.
    """
    if node.type == "arrow_function" and node.parent and node.parent.type == "variable_declarator":
        name_node = node.parent.child_by_field_name("name")
        if name_node:
            return node_text(name_node, source).strip()
    for field in (
        "name", "declarator", "declarator_name", "function",
        "type", "alias", "identifier",
    ):
        child = node.child_by_field_name(field)
        if child is not None:
            text = node_text(child, source).strip()
            # Declarators can be nested in C/C++; use the last identifier.
            identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
            if identifiers:
                return identifiers[-1]

    # Common grammar fallback: first identifier descendant.
    stack = list(node.named_children)
    while stack:
        current = stack.pop(0)
        if current.type in {
            "identifier", "type_identifier", "field_identifier",
            "property_identifier", "namespace_identifier",
            "qualified_identifier", "simple_identifier",
        }:
            return node_text(current, source).strip()
        stack[0:0] = list(current.named_children)

    return None


def qualified_name(node, source: bytes) -> Optional[str]:
    """
    Preserve qualified names such as:
      foo.bar()
      std::vector
      package.Class.method
    """
    text = node_text(node, source).strip()
    if not text:
        return None

    # Keep a conservative useful representation.
    text = re.sub(r"\s+", "", text)
    match = re.search(
        r"([A-Za-z_][A-Za-z0-9_]*(?:(?:::|\.|->)[A-Za-z_][A-Za-z0-9_]*)*)$",
        text,
    )
    return match.group(1) if match else text


def find_identifier_descendant(node, source: bytes):
    stack = list(node.named_children)
    while stack:
        current = stack.pop(0)
        if current.type in {
            "identifier", "type_identifier", "field_identifier",
            "property_identifier", "simple_identifier",
        }:
            return node_text(current, source).strip()
        stack[0:0] = list(current.named_children)
    return None


def extract_parameters(node, source: bytes) -> list[str]:
    params = []
    for child in node.named_children:
        if child.type in PARAMETER_NODES:
            for p in child.named_children:
                text = node_text(p, source).strip()
                if text:
                    params.append(text)
    return params


def is_jsx_node(node):
    return node.type in {"jsx_element", "jsx_self_closing_element"}


# ============================================================================
# LANGUAGE-SPECIFIC IMPORT / PACKAGE EXTRACTION
# ============================================================================

def parse_import_text(statement: str, lang: str):
    """
    Normalize imports/includes into:
      module, imported, local, kind
    """
    result = []
    text = statement.strip()

    if lang in {"javascript", "typescript"}:
        side = re.match(r'import\s+[\'"]([^\'"]+)[\'"]', text)
        if side:
            return [{
                "module": side.group(1),
                "imported": None,
                "local": None,
                "kind": "side_effect",
            }]

        m = re.search(
            r'import\s+(.*?)\s+from\s+[\'"]([^\'"]+)[\'"]',
            text,
            re.DOTALL,
        )
        if not m:
            return []

        clause, module_name = m.group(1).strip(), m.group(2)

        ns = re.search(r"\*\s+as\s+([A-Za-z_$][\w$]*)", clause)
        if ns:
            result.append({
                "module": module_name, "imported": "*",
                "local": ns.group(1), "kind": "namespace",
            })

        named = re.search(r"\{(.*?)\}", clause, re.DOTALL)
        if named:
            for item in named.group(1).split(","):
                item = re.sub(r"^\s*type\s+", "", item.strip())
                if not item:
                    continue
                if re.search(r"\s+as\s+", item):
                    imported, local = re.split(r"\s+as\s+", item, maxsplit=1)
                else:
                    imported = local = item
                result.append({
                    "module": module_name,
                    "imported": imported.strip(),
                    "local": local.strip(),
                    "kind": "named",
                })

        default_part = clause
        if named:
            default_part = clause[:named.start()].strip().rstrip(",")
        if default_part and not default_part.startswith("*"):
            result.append({
                "module": module_name,
                "imported": "default",
                "local": default_part,
                "kind": "default",
            })
        return result

    if lang in {"c", "cpp"}:
        m = re.search(r'#\s*include\s*[<"]([^>"]+)[>"]', text)
        if m:
            return [{
                "module": m.group(1),
                "imported": None,
                "local": None,
                "kind": "include",
            }]
        return []

    if lang == "python":
        m = re.match(r"^\s*import\s+(.+)", text, re.DOTALL)
        if m:
            for item in m.group(1).split(","):
                item = item.strip()
                parts = re.split(r"\s+as\s+", item, maxsplit=1)
                result.append({
                    "module": parts[0],
                    "imported": "*",
                    "local": parts[-1],
                    "kind": "import",
                })
            return result

        m = re.match(
            r"^\s*from\s+([A-Za-z0-9_.$]+)\s+import\s+(.+)",
            text,
            re.DOTALL,
        )
        if m:
            module_name = m.group(1)
            for item in m.group(2).split(","):
                item = item.strip()
                parts = re.split(r"\s+as\s+", item, maxsplit=1)
                result.append({
                    "module": module_name,
                    "imported": parts[0],
                    "local": parts[-1],
                    "kind": "from_import",
                })
            return result
        return []

    if lang in {"java", "kotlin"}:
        m = re.search(r"\bimport\s+(?:static\s+)?([A-Za-z0-9_.$*]+)", text)
        if m:
            module = m.group(1)
            local = module.split(".")[-1]
            return [{
                "module": module,
                "imported": local,
                "local": local,
                "kind": "import",
            }]
        return []

    if lang == "csharp":
        m = re.search(r"\busing\s+(?:static\s+)?([A-Za-z0-9_.]+)", text)
        if m:
            module = m.group(1)
            return [{
                "module": module,
                "imported": module.split(".")[-1],
                "local": module.split(".")[-1],
                "kind": "using",
            }]
        return []

    if lang == "go":
        # import "fmt"
        m = re.search(r'\bimport\s+(?:"([^"]+)"|`([^`]+)`)', text)
        if m:
            module = m.group(1) or m.group(2)
            return [{
                "module": module,
                "imported": module.split("/")[-1],
                "local": module.split("/")[-1],
                "kind": "import",
            }]

        # import alias "path"
        m = re.search(r'\bimport\s+([A-Za-z_][A-Za-z0-9_]*)\s+"([^"]+)"', text)
        if m:
            return [{
                "module": m.group(2),
                "imported": m.group(1),
                "local": m.group(1),
                "kind": "import",
            }]
        return []

    return []


def extract_imports_from_tree(root, source: bytes, lang: str):
    imports = []

    def walk(node):
        if node.type in IMPORT_NODES:
            parsed = parse_import_text(node_text(node, source), lang)
            for item in parsed:
                imports.append({
                    **item,
                    "line": line(node),
                    "source": node_text(node, source),
                })
        for child in node.named_children:
            walk(child)

    walk(root)

    # SQL does not have normal module imports.
    if lang == "sql":
        return []

    return imports


# ============================================================================
# SQL EXTRACTION
# ============================================================================

SQL_TABLE_RE = re.compile(
    r"\b(?:from|join|update|into|delete\s+from|truncate\s+table|"
    r"create\s+table|alter\s+table)\s+([A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*)?)",
    re.IGNORECASE,
)


def extract_sql_objects(source_text: str):
    tables = []
    for match in SQL_TABLE_RE.finditer(source_text):
        table = match.group(1)
        if table.lower() not in {x.lower() for x in tables}:
            tables.append(table)

    statements = []
    for match in re.finditer(
        r"\b(select|insert|update|delete|create|alter|drop|truncate|merge)\b",
        source_text,
        re.IGNORECASE,
    ):
        statements.append({
            "kind": match.group(1).upper(),
            "line": source_text[:match.start()].count("\n") + 1,
        })

    return tables, statements


# ============================================================================
# EXPORT / PACKAGE / NAMESPACE EXTRACTION
# ============================================================================

def extract_module_metadata(source_text: str, lang: str):
    exports = []
    namespaces = []

    if lang in {"javascript", "typescript"}:
        patterns = [
            r"\bexport\s+default\s+(?:function|class)\s+([A-Za-z_$][\w$]*)",
            r"\bexport\s+(?:async\s+)?function\s+([A-Za-z_$][\w$]*)",
            r"\bexport\s+class\s+([A-Za-z_$][\w$]*)",
            r"\bexport\s+(?:const|let|var)\s+([A-Za-z_$][\w$]*)",
        ]
        for pattern in patterns:
            exports.extend(m.group(1) for m in re.finditer(pattern, source_text))

        for block in re.findall(r"\bexport\s*\{(.*?)\}", source_text, re.DOTALL):
            for item in block.split(","):
                item = item.strip()
                if item:
                    exports.append(item)

    elif lang == "python":
        m = re.search(r"^\s*__all__\s*=\s*\[(.*?)\]", source_text, re.M | re.S)
        if m:
            exports.extend(re.findall(r"['\"]([^'\"]+)['\"]", m.group(1)))

    elif lang in {"java", "kotlin"}:
        namespaces.extend(
            re.findall(r"\bpackage\s+([A-Za-z0-9_.]+)", source_text)
        )

    elif lang == "csharp":
        namespaces.extend(
            re.findall(r"\bnamespace\s+([A-Za-z0-9_.]+)", source_text)
        )

    elif lang == "go":
        namespaces.extend(
            re.findall(r"\bpackage\s+([A-Za-z0-9_]+)", source_text)
        )

    return sorted(set(exports)), sorted(set(namespaces))


# ============================================================================
# DEFINITION EXTRACTION
# ============================================================================

def definition_label(lang: str, node_type: str, name: str = "") -> str:
    if node_type in CLASS_NODES:
        if node_type in {"interface_declaration", "trait_declaration"}:
            return "Interface"
        if node_type in {"enum_declaration", "enum_specifier"}:
            return "Enum"
        if node_type in {"struct_specifier", "struct_declaration"}:
            return "Struct"
        return "Class"

    if node_type in FUNCTION_NODES:
        if lang in {"javascript", "typescript"} and name and (name[:1].isupper() or name.startswith("use")):
            return "Component"
        return "Function"

    # Language-specific declarations which represent named symbols.
    if node_type in {
        "type_declaration", "type_alias_declaration",
        "type_definition", "type_spec",
    }:
        return "Type"

    if node_type in {
        "const_declaration", "const_spec", "var_declaration",
        "variable_declaration", "lexical_declaration",
    }:
        return "Variable"

    return "Symbol"


def looks_like_definition(node) -> bool:
    return (
        node.type in FUNCTION_NODES
        or node.type in CLASS_NODES
        or node.type in {
            "type_declaration", "type_alias_declaration",
            "type_definition", "type_spec",
            "const_declaration", "const_spec", "var_declaration",
        }
    )


def extract_definitions(root, source: bytes, lang: str):
    definitions = {}

    def walk(node, parent_symbol=None):
        # JavaScript/TypeScript variable-based functions/components:
        #   const App = () => {}
        #   const helper = function () {}
        #   const App = memo(() => {})
        if (
            lang in {"javascript", "typescript"}
            and node.type == "variable_declarator"
        ):
            name_node = node.child_by_field_name("name")
            value_node = node.child_by_field_name("value")

            if name_node is not None and value_node is not None:
                name = node_text(name_node, source).strip()
                is_function_value = value_node.type in {
                    "arrow_function",
                    "function_expression",
                }

                if value_node.type == "call_expression":
                    fn = value_node.child_by_field_name("function")
                    wrapper = (
                        node_text(fn, source).strip()
                        if fn is not None else ""
                    )
                    is_function_value = wrapper in {
                        "memo", "forwardRef", "lazy",
                    }

                if is_function_value and name:
                    label = (
                        "Component"
                        if name[:1].isupper() or name.startswith("use")
                        else "Function"
                    )
                    definition_id = stable_id(
                        label, current_filepath, name, str(node.start_byte)
                    )

                    graph.add_node(
                        label,
                        definition_id,
                        name=name,
                        path=current_filepath,
                        language=lang,
                        line=line(node),
                        column=column(node),
                        declaration_type=value_node.type,
                        parameters=extract_parameters(value_node, source),
                    )

                    graph.add_relationship(
                        current_file_id,
                        "DEFINES",
                        definition_id,
                        line=line(node),
                    )

                    definitions.setdefault(name, {
                        "id": definition_id,
                        "label": label,
                        "name": name,
                        "line": line(node),
                        "node": node,
                    })

                    parent_symbol = definition_id

        if looks_like_definition(node):
            name = node_name(node, source)

            # For C/C++ declarators, node_name can fail on a complex declaration.
            if not name:
                name = find_identifier_descendant(node, source)

            if name:
                label = definition_label(lang, node.type, name)
                definition_id = stable_id(
                    label, current_filepath, name, str(node.start_byte)
                )

                graph.add_node(
                    label,
                    definition_id,
                    name=name,
                    path=current_filepath,
                    language=lang,
                    line=line(node),
                    column=column(node),
                    declaration_type=node.type,
                    parameters=extract_parameters(node, source),
                )

                graph.add_relationship(
                    current_file_id,
                    "DEFINES",
                    definition_id,
                    line=line(node),
                )

                definitions.setdefault(name, {
                    "id": definition_id,
                    "label": label,
                    "name": name,
                    "line": line(node),
                    "node": node,
                })

                parent_symbol = definition_id

        for child in node.named_children:
            walk(child, parent_symbol)

    # graph/current IDs are set by parse_file.
    graph = _active_graph
    current_filepath = _active_filepath
    current_file_id = _active_file_id

    walk(root)
    return definitions


# ============================================================================
# CALL EXTRACTION
# ============================================================================

def extract_calls(root, source: bytes, lang: str, definitions):
    calls = []

    def walk(node, current_scope=None):
        nonlocal calls

        if node.type in FUNCTION_NODES:
            name = node_name(node, source)
            if name and name in definitions:
                current_scope = definitions[name]["id"]

        if node.type in CALL_NODES | CONTROL_CALL_NODES:
            callee_node = (
                node.child_by_field_name("function")
                or node.child_by_field_name("name")
                or node.child_by_field_name("method")
                or node.child_by_field_name("constructor")
            )

            callee = (
                qualified_name(callee_node, source)
                if callee_node is not None
                else find_identifier_descendant(node, source)
            )

            if callee:
                call_id = stable_id(
                    "CallSite",
                    current_filepath,
                    callee,
                    str(node.start_byte),
                )

                graph.add_node(
                    "CallSite",
                    call_id,
                    name=callee,
                    file=current_filepath,
                    language=lang,
                    line=line(node),
                    column=column(node),
                    source=node_text(node, source),
                )

                owner = current_scope or current_file_id
                graph.add_relationship(
                    owner,
                    "CALL_SITE",
                    call_id,
                    line=line(node),
                )

                calls.append((call_id, callee, current_scope))

        for child in node.named_children:
            walk(child, current_scope)

    graph = _active_graph
    current_filepath = _active_filepath
    current_file_id = _active_file_id
    walk(root)
    return calls


# ============================================================================
# JAVASCRIPT / TYPESCRIPT JSX EXTRACTION
# ============================================================================

def extract_jsx(root, source: bytes, definitions):
    def jsx_name(opening):
        name_node = opening.child_by_field_name("name")
        return node_text(name_node, source) if name_node else None

    def props(opening, render_id):
        for child in opening.children:
            if child.type != "jsx_attribute":
                continue

            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if not name_node:
                continue

            prop_name = node_text(name_node, source)
            if value_node:
                raw_value = node_text(value_node, source)
                value_kind = (
                    "literal"
                    if value_node.type == "string"
                    else "expression"
                )
            else:
                raw_value = "true"
                value_kind = "boolean"

            prop_id = stable_id(
                "PropBinding",
                current_filepath,
                prop_name,
                f"{render_id}|{child.start_byte}",
            )

            graph.add_node(
                "PropBinding",
                prop_id,
                name=prop_name,
                value=raw_value,
                value_kind=value_kind,
                file=current_filepath,
                line=line(child),
                column=column(child),
                source=node_text(child, source),
            )

            graph.add_relationship(
                render_id,
                "PASSES_PROP",
                prop_id,
                line=line(child),
                name=prop_name,
                value=raw_value,
            )

    def walk(node, current_scope=None):
        if node.type in FUNCTION_NODES:
            name = node_name(node, source)
            if name in definitions:
                current_scope = definitions[name]["id"]

        if is_jsx_node(node):
            opening = node
            if node.type == "jsx_element":
                opening = next(
                    (
                        child for child in node.named_children
                        if child.type == "jsx_opening_element"
                    ),
                    None,
                )

            if opening:
                rendered_name = jsx_name(opening)
                if rendered_name and (
                    rendered_name[:1].isupper() or "." in rendered_name
                ):
                    render_id = stable_id(
                        "RenderSite",
                        current_filepath,
                        rendered_name,
                        str(node.start_byte),
                    )

                    graph.add_node(
                        "RenderSite",
                        render_id,
                        name=rendered_name,
                        file=current_filepath,
                        language=language_of(current_filepath),
                        line=line(node),
                        column=column(node),
                        source=node_text(node, source),
                        raw_target=rendered_name,
                    )

                    owner = current_scope or current_file_id
                    graph.add_relationship(
                        owner, "RENDERS", render_id, line=line(node)
                    )
                    props(opening, render_id)

        for child in node.named_children:
            walk(child, current_scope)

    graph = _active_graph
    current_filepath = _active_filepath
    current_file_id = _active_file_id
    walk(root)


# ============================================================================
# FILE PARSER
# ============================================================================

# These globals are only used to keep the language-specific walkers concise.
# parse_file is called sequentially, so they are never shared between threads.
_active_graph = None
_active_filepath = None
_active_file_id = None


def parse_file(filepath: str, target_root: Path):
    global _active_graph, _active_filepath, _active_file_id

    filepath = normalize_path(filepath)
    absolute_path = target_root / filepath
    parser = get_parser(filepath)
    lang = language_of(filepath)

    if absolute_path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError(
            f"File exceeds GRAPH_RAG_MAX_FILE_SIZE_MB={MAX_FILE_SIZE_MB}: "
            f"{filepath}"
        )

    with open(absolute_path, "rb") as f:
        source = f.read()

    tree = parser.parse(source)
    graph = GraphIR()

    file_id = stable_id("File", filepath, filepath)
    graph.add_node(
        "File",
        file_id,
        path=filepath,
        name=Path(filepath).name,
        extension=extension_of(filepath),
        language=lang,
        size_bytes=len(source),
    )

    _active_graph = graph
    _active_filepath = filepath
    _active_file_id = file_id

    definitions = extract_definitions(tree.root_node, source, lang)
    imports = extract_imports_from_tree(tree.root_node, source, lang)

    # Imports/includes.
    module_ids = {}
    for index, item in enumerate(imports):
        module_name = item["module"]
        import_id = stable_id(
            "Import", filepath, module_name, str(index)
        )

        graph.add_node(
            "Import",
            import_id,
            module=module_name,
            local=item.get("local"),
            imported=item.get("imported"),
            kind=item.get("kind"),
            file=filepath,
            language=lang,
            line=item["line"],
            source=item.get("source", ""),
        )
        graph.add_relationship(
            file_id, "IMPORTS", import_id, line=item["line"]
        )

        if module_name not in module_ids:
            module_id = stable_id("Module", filepath, module_name)
            module_ids[module_name] = module_id
            graph.add_node(
                "Module",
                module_id,
                name=module_name,
                importer=filepath,
                language=lang,
            )
            graph.add_relationship(
                file_id, "USES_MODULE", module_id
            )

    # SQL-specific structure.
    source_text = source.decode("utf-8", errors="replace")
    if lang == "sql":
        tables, statements = extract_sql_objects(source_text)

        for table in tables:
            table_id = stable_id("Table", filepath, table)
            graph.add_node(
                "Table",
                table_id,
                name=table,
                file=filepath,
                language="sql",
            )
            graph.add_relationship(file_id, "REFERENCES_TABLE", table_id)

        for index, stmt in enumerate(statements):
            statement_id = stable_id(
                "SQLStatement", filepath, stmt["kind"], str(index)
            )
            graph.add_node(
                "SQLStatement",
                statement_id,
                kind=stmt["kind"],
                file=filepath,
                language="sql",
                line=stmt["line"],
            )
            graph.add_relationship(
                file_id,
                "CONTAINS_SQL",
                statement_id,
                line=stmt["line"],
            )

    # Calls.
    calls = extract_calls(tree.root_node, source, lang, definitions)

    # JSX/React component rendering.
    if lang in {"javascript", "typescript"} and extension_of(filepath) in {
        ".jsx", ".tsx"
    }:
        extract_jsx(tree.root_node, source, definitions)

    exports, namespaces = extract_module_metadata(source_text, lang)

    graph.nodes[file_id]["definitions"] = list(definitions)
    graph.nodes[file_id]["imports"] = imports
    graph.nodes[file_id]["exports"] = exports
    graph.nodes[file_id]["namespaces"] = namespaces

    for namespace in namespaces:
        namespace_id = stable_id("Namespace", filepath, namespace)
        graph.add_node(
            "Namespace",
            namespace_id,
            name=namespace,
            file=filepath,
            language=lang,
        )
        graph.add_relationship(file_id, "BELONGS_TO", namespace_id)

    # Reset context.
    _active_graph = None
    _active_filepath = None
    _active_file_id = None

    return {
        "graph": graph,
        "filepath": filepath,
        "language": lang,
        "definitions": definitions,
        "imports": imports,
        "exports": exports,
    }


# ============================================================================
# LOCAL MODULE / INCLUDE RESOLUTION
# ============================================================================

def resolve_module(source_file: str, module_name: str, target_root: Path):
    """
    Resolve only local/project references.

    External packages are deliberately not guessed.
    """
    if not module_name:
        return None

    source_dir = (target_root / Path(source_file).parent).resolve()
    raw = module_name.replace("\\", "/")

    # JS/TS relative imports.
    if raw.startswith("."):
        base = (source_dir / raw).resolve()
        candidates = [
            base,
            *[Path(str(base) + ext) for ext in SUPPORTED_EXTENSIONS],
        ]
        candidates.extend(
            base / f"index{ext}"
            for ext in SUPPORTED_EXTENSIONS
        )
        return first_relative_candidate(candidates, target_root)

    # C/C++ include paths: resolve only a file that exists inside target_root.
    if extension_of(source_file) in {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh"}:
        candidate = (source_dir / raw).resolve()
        return first_relative_candidate(
            [candidate, target_root / raw],
            target_root,
        )

    # Python local modules.
    if language_of(source_file) == "python":
        parts = raw.split(".")
        base = target_root.joinpath(*parts)
        return first_relative_candidate(
            [
                base.with_suffix(".py"),
                base / "__init__.py",
            ],
            target_root,
        )

    # Java/Kotlin/C# packages are project dependent. Resolve by filename only
    # when an unambiguous source file exists.
    if language_of(source_file) in {"java", "kotlin", "csharp"}:
        simple = raw.split(".")[-1]
        candidates = list(target_root.rglob(f"{simple}.java"))
        candidates += list(target_root.rglob(f"{simple}.kt"))
        candidates += list(target_root.rglob(f"{simple}.cs"))
        if len(candidates) == 1:
            return normalize_path(str(candidates[0].relative_to(target_root)))

    return None


def first_relative_candidate(candidates, target_root: Path):
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(target_root.resolve())
        except (ValueError, OSError):
            continue

        if resolved.exists() and resolved.is_file():
            return normalize_path(str(resolved.relative_to(target_root)))

    return None


# ============================================================================
# GLOBAL SYMBOL INDEX
# ============================================================================

def build_symbol_index(parsed_files):
    by_file_and_name = {}
    definitions_by_name = defaultdict(list)

    for item in parsed_files:
        filepath = item["filepath"]
        for name, definition in item["definitions"].items():
            by_file_and_name[(filepath, name)] = definition
            definitions_by_name[name].append((filepath, definition))

    return by_file_and_name, definitions_by_name


# ============================================================================
# CROSS-FILE RESOLUTION
# ============================================================================

def link_imports(parsed_files, all_graph, symbol_index, definitions_by_name, target_root):
    for item in parsed_files:
        filepath = item["filepath"]
        imports = item["imports"]
        lang = item["language"]

        for imp in imports:
            module_name = imp.get("module")
            target_file = resolve_module(
                filepath, module_name, target_root
            )

            module_id = stable_id("Module", filepath, module_name)
            if target_file:
                target_file_id = stable_id(
                    "File", target_file, target_file
                )

                if target_file_id not in all_graph.nodes:
                    all_graph.add_node(
                        "File",
                        target_file_id,
                        path=target_file,
                        name=Path(target_file).name,
                        extension=extension_of(target_file),
                        language=language_of(target_file),
                    )

                all_graph.add_relationship(
                    module_id,
                    "RESOLVES_TO",
                    target_file_id,
                    resolution="local_project",
                )

                imported_name = imp.get("imported")
                local_name = imp.get("local")

                if imported_name and imported_name not in {"*", "default"}:
                    target = symbol_index.get(
                        (target_file, imported_name)
                    )
                    if target:
                        source_file_id = stable_id(
                            "File", filepath, filepath
                        )
                        all_graph.add_relationship(
                            source_file_id,
                            "IMPORTS_SYMBOL",
                            target["id"],
                            symbol=imported_name,
                            resolution="local_symbol",
                        )
                        continue

                # For Java/Kotlin/C# and C/C++, importing a file/package does
                # not necessarily identify one symbol. Keep the file edge.
                continue

            # No confident target: preserve evidence without guessing.
            unresolved_id = stable_id(
                "UnresolvedSymbol",
                filepath,
                module_name,
            )
            if unresolved_id not in all_graph.nodes:
                all_graph.add_node(
                    "UnresolvedSymbol",
                    unresolved_id,
                    name=module_name,
                    source_file=filepath,
                    language=lang,
                    reason="local_reference_not_resolved",
                )
            all_graph.add_relationship(
                stable_id("File", filepath, filepath),
                "UNRESOLVED_IMPORT",
                unresolved_id,
                module=module_name,
            )


def link_calls(parsed_files, all_graph):
    for item in parsed_files:
        filepath = item["filepath"]
        definitions = item["definitions"]

        for call_id, node in list(item["graph"].nodes.items()):
            if node["label"] != "CallSite":
                continue

            callee = node["name"]
            # Exact local-name resolution only.
            simple = re.split(r"(?:::|\.|->)", callee)[-1]

            if simple in definitions:
                all_graph.add_relationship(
                    call_id,
                    "CALLS",
                    definitions[simple]["id"],
                    resolution="local_exact",
                )
            else:
                unresolved_id = stable_id(
                    "UnresolvedSymbol",
                    filepath,
                    callee,
                    str(node["line"]),
                )
                if unresolved_id not in all_graph.nodes:
                    all_graph.add_node(
                        "UnresolvedSymbol",
                        unresolved_id,
                        name=callee,
                        source_file=filepath,
                        language=item["language"],
                        line=node["line"],
                        reason="call_target_not_resolved",
                    )
                all_graph.add_relationship(
                    call_id,
                    "UNRESOLVED_CALL",
                    unresolved_id,
                )


# ============================================================================
# BUILD PROJECT
# ============================================================================

def build_project(target_directory="data"):
    target_root = Path(target_directory).resolve()

    if not target_root.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_root}")

    print(f"\nScanning: {target_root}")
    print("Supported languages:", ", ".join(sorted(set(LANGUAGE_BY_EXTENSION.values()))))

    parsed_files = []
    skipped = 0

    for root, dirs, files in os.walk(target_root):
        dirs[:] = [
            d for d in dirs
            if d not in IGNORED_DIRECTORIES
        ]

        for filename in files:
            suffix = extension_of(filename)
            if suffix not in SUPPORTED_EXTENSIONS:
                continue

            absolute_file = Path(root) / filename
            relative_file = normalize_path(
                str(absolute_file.relative_to(target_root))
            )

            print(f"  Parsing [{language_of(relative_file)}] {relative_file}")

            try:
                parsed_files.append(
                    parse_file(relative_file, target_root)
                )
            except Exception as exc:
                skipped += 1
                print(f"  ERROR: {relative_file}")
                print(f"         {type(exc).__name__}: {exc}")

    if not parsed_files:
        raise RuntimeError(
            "No supported source files were successfully parsed."
        )

    print("\nBuilding global symbol index...")
    symbol_index, definitions_by_name = build_symbol_index(parsed_files)

    all_graph = GraphIR()

    for item in parsed_files:
        all_graph.nodes.update(item["graph"].nodes)
        all_graph.relationships.extend(item["graph"].relationships)

    print("Resolving local imports/includes/packages...")
    link_imports(
        parsed_files,
        all_graph,
        symbol_index,
        definitions_by_name,
        target_root,
    )

    print("Resolving local call sites...")
    link_calls(parsed_files, all_graph)

    # De-duplicate relationships.
    unique_relationships = {}
    for rel in all_graph.relationships:
        key = (
            rel["source"],
            rel["type"],
            rel["target"],
        )
        unique_relationships[key] = rel

    all_graph.relationships = list(unique_relationships.values())

    print("Writing graph to Neo4j...")
    write_to_neo4j(all_graph)

    labels = defaultdict(int)
    languages = defaultdict(int)

    for node in all_graph.nodes.values():
        labels[node["label"]] += 1
        if node.get("language"):
            languages[node["language"]] += 1

    print("\n======================================")
    print("MULTI-LANGUAGE AST GRAPH BUILD COMPLETE")
    print("======================================")
    print(f"Files:            {labels['File']}")
    print(f"Classes/structs:  {labels['Class'] + labels['Struct'] + labels['Interface'] + labels['Enum']}")
    print(f"Functions:        {labels['Function']}")
    print(f"Variables:        {labels['Variable']}")
    print(f"Types:             {labels['Type']}")
    print(f"Calls:             {labels['CallSite']}")
    print(f"Imports:           {labels['Import']}")
    print(f"Modules:           {labels['Module']}")
    print(f"Namespaces:        {labels['Namespace']}")
    print(f"SQL tables:        {labels['Table']}")
    print(f"SQL statements:    {labels['SQLStatement']}")
    print(f"Unresolved:        {labels['UnresolvedSymbol']}")
    print(f"Relationships:     {len(all_graph.relationships)}")
    print(f"Skipped files:     {skipped}")
    print("\nFiles by language:")
    for lang, count in sorted(languages.items()):
        print(f"  {lang:<14} {count}")

    return all_graph


# ============================================================================
# NEO4J SCHEMA + WRITE
# ============================================================================

def write_to_neo4j(graph: GraphIR):
    with driver.session() as session:
        print("\nClearing previous graph...")
        session.run("MATCH (n) DETACH DELETE n")

        labels = sorted({
            node["label"] for node in graph.nodes.values()
        })

        for label in labels:
            try:
                session.run(
                    f"""
                    CREATE CONSTRAINT IF NOT EXISTS
                    FOR (n:`{label}`)
                    REQUIRE n.id IS UNIQUE
                    """
                )
            except Exception as exc:
                print(f"Schema warning for {label}: {exc}")

        nodes_by_label = defaultdict(list)

        for node in graph.nodes.values():
            properties = {
                key: value
                for key, value in node.items()
                if key != "label"
                and not key.startswith("raw_")
                and key not in {"definitions", "imports", "exports", "namespaces"}
            }

            # Neo4j properties cannot contain arbitrary nested dictionaries.
            # Convert unusual values to strings while keeping useful lists.
            for key, value in list(properties.items()):
                if isinstance(value, dict):
                    properties[key] = str(value)

            nodes_by_label[node["label"]].append(properties)

        print("Writing nodes...")
        for label, rows in nodes_by_label.items():
            for batch_start in range(0, len(rows), 1000):
                batch = rows[batch_start:batch_start + 1000]
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MERGE (n:`{label}` {{id: row.id}})
                    SET n += row
                    """,
                    rows=batch,
                )

        relationships_by_type = defaultdict(list)
        for rel in graph.relationships:
            relationships_by_type[rel["type"]].append(rel)

        print("Writing relationships...")
        for relation, rows in relationships_by_type.items():
            for batch_start in range(0, len(rows), 1000):
                batch = rows[batch_start:batch_start + 1000]
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (a {{id: row.source}})
                    MATCH (b {{id: row.target}})
                    MERGE (a)-[r:`{relation}`]->(b)
                    SET r += row.properties
                    """,
                    rows=[
                        {
                            "source": row["source"],
                            "target": row["target"],
                            "properties": row["properties"],
                        }
                        for row in batch
                    ],
                )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        driver.verify_connectivity()
        print("Connected to Neo4j.")
        build_project(os.getenv("GRAPH_RAG_TARGET", "data"))
    finally:
        driver.close()
