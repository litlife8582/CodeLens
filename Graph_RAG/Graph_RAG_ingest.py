import os
import re
import hashlib
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from neo4j import GraphDatabase

import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv(override=True)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_URI or not NEO4J_PASSWORD:
    raise ValueError(
        "Missing NEO4J_URI or NEO4J_PASSWORD in .env"
    )

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)


# ============================================================
# TREE-SITTER PARSERS
# ============================================================

JS_LANGUAGE = Language(tsjs.language())
TS_LANGUAGE = Language(tsts.language_typescript())
TSX_LANGUAGE = Language(tsts.language_tsx())

JS_PARSER = Parser(JS_LANGUAGE)
TS_PARSER = Parser(TS_LANGUAGE)
TSX_PARSER = Parser(TSX_LANGUAGE)


SUPPORTED_EXTENSIONS = {
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
}

IGNORED_DIRECTORIES = {
    "node_modules",
    ".git",
    "dist",
    "build",
    ".next",
    "coverage",
    "out",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_path(path: str) -> str:
    return Path(path).as_posix()


def stable_id(kind: str, path: str, name: str, location: str = "") -> str:
    """
    Stable identifier.

    Example:

        Component + components/Button.jsx + Button

    will always produce the same ID.
    """

    raw = f"{kind}|{path}|{name}|{location}"

    return hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()


def node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode(
        "utf-8",
        errors="replace"
    )


def line(node) -> int:
    return node.start_point[0] + 1


def column(node) -> int:
    return node.start_point[1] + 1


def is_component_name(name: str) -> bool:
    return bool(name) and (
        name[0].isupper()
        or name.startswith("use")
    )


def get_parser(filepath: str):

    suffix = Path(filepath).suffix.lower()

    if suffix == ".tsx":
        return TSX_PARSER

    if suffix == ".ts":
        return TS_PARSER

    return JS_PARSER


# ============================================================
# IMPORT PARSING
# ============================================================

def parse_import_statement(source_text: str):

    """
    Parse common ES module import syntax.

    Examples:

        import Button from "./Button";

        import { Button } from "./Button";

        import { Button as MyButton } from "./Button";

        import * as UI from "./ui";

        import React from "react";

        import "./styles.css";
    """

    result = []

    # --------------------------------------------------------
    # Side-effect import
    # --------------------------------------------------------

    side_effect = re.match(
        r'^\s*import\s+[\'"]([^\'"]+)[\'"]',
        source_text
    )

    if side_effect:
        return [{
            "local": None,
            "imported": None,
            "module": side_effect.group(1),
            "kind": "side_effect",
        }]

    # --------------------------------------------------------
    # Normal import
    # --------------------------------------------------------

    match = re.search(
        r'import\s+(.*?)\s+from\s+[\'"]([^\'"]+)[\'"]',
        source_text,
        re.DOTALL
    )

    if not match:
        return result

    clause = match.group(1).strip()
    module_name = match.group(2)

    # --------------------------------------------------------
    # namespace:
    #
    # import * as UI from "./ui"
    # --------------------------------------------------------

    namespace_match = re.search(
        r'\*\s+as\s+([A-Za-z_$][\w$]*)',
        clause
    )

    if namespace_match:

        result.append({
            "local": namespace_match.group(1),
            "imported": "*",
            "module": module_name,
            "kind": "namespace",
        })

        return result

    # --------------------------------------------------------
    # named imports:
    #
    # import { Button, Card as MyCard }
    # --------------------------------------------------------

    named_match = re.search(
        r'\{(.*?)\}',
        clause,
        re.DOTALL
    )

    if named_match:

        named_part = named_match.group(1)

        for item in named_part.split(","):

            item = item.strip()

            if not item:
                continue

            item = re.sub(
                r"^type\s+",
                "",
                item
            )

            if re.search(r"\s+as\s+", item):

                imported_name, local_name = re.split(
                    r"\s+as\s+",
                    item,
                    maxsplit=1
                )

            else:

                imported_name = item
                local_name = item

            result.append({
                "local": local_name.strip(),
                "imported": imported_name.strip(),
                "module": module_name,
                "kind": "named",
            })

    # --------------------------------------------------------
    # default import
    #
    # import Button from "./Button"
    #
    # import Button, { X } from "./Button"
    # --------------------------------------------------------

    default_part = clause

    if named_match:
        default_part = clause[:named_match.start()].strip()

    default_part = default_part.rstrip(",")

    if (
        default_part
        and not default_part.startswith("{")
        and not default_part.startswith("*")
    ):

        result.append({
            "local": default_part,
            "imported": "default",
            "module": module_name,
            "kind": "default",
        })

    return result


# ============================================================
# EXPORT PARSING
# ============================================================

def parse_exports(source_text: str):

    exports = []

    # --------------------------------------------------------
    # export default function Foo
    # --------------------------------------------------------

    match = re.search(
        r'export\s+default\s+function\s+([A-Za-z_$][\w$]*)',
        source_text
    )

    if match:

        exports.append({
            "local": match.group(1),
            "exported": "default",
        })

    # --------------------------------------------------------
    # export function Foo
    # --------------------------------------------------------

    for match in re.finditer(
        r'export\s+(?:async\s+)?function\s+([A-Za-z_$][\w$]*)',
        source_text
    ):

        exports.append({
            "local": match.group(1),
            "exported": match.group(1),
        })

    # --------------------------------------------------------
    # export const Foo
    # export let Foo
    # --------------------------------------------------------

    for match in re.finditer(
        r'export\s+(?:const|let|var)\s+([A-Za-z_$][\w$]*)',
        source_text
    ):

        exports.append({
            "local": match.group(1),
            "exported": match.group(1),
        })

    # --------------------------------------------------------
    # export default Foo
    # --------------------------------------------------------

    match = re.search(
        r'export\s+default\s+([A-Za-z_$][\w$]*)',
        source_text
    )

    if match:

        name = match.group(1)

        if name not in {"function", "class"}:

            exports.append({
                "local": name,
                "exported": "default",
            })

    # --------------------------------------------------------
    # export { Foo, Bar as Baz }
    # --------------------------------------------------------

    for block in re.findall(
        r'export\s*\{(.*?)\}',
        source_text,
        re.DOTALL
    ):

        for item in block.split(","):

            item = item.strip()

            if not item:
                continue

            if re.search(r"\s+as\s+", item):

                local_name, exported_name = re.split(
                    r"\s+as\s+",
                    item,
                    maxsplit=1
                )

            else:

                local_name = item
                exported_name = item

            exports.append({
                "local": local_name.strip(),
                "exported": exported_name.strip(),
            })

    return exports


# ============================================================
# GRAPH MODEL
# ============================================================

class GraphIR:

    def __init__(self):

        self.nodes = {}
        self.relationships = []

    def add_node(
        self,
        label,
        node_id,
        **properties
    ):

        self.nodes[node_id] = {
            "label": label,
            "id": node_id,
            **properties
        }

    def add_relationship(
        self,
        source_id,
        relation,
        target_id,
        **properties
    ):

        self.relationships.append({
            "source": source_id,
            "type": relation,
            "target": target_id,
            "properties": properties,
        })


# ============================================================
# PARSE ONE FILE
# ============================================================

def parse_file(
    filepath: str,
    target_root: Path
):

    filepath = normalize_path(filepath)

    parser = get_parser(filepath)

    absolute_path = target_root / filepath

    with open(
        absolute_path,
        "rb"
    ) as f:

        source = f.read()

    tree = parser.parse(source)

    graph = GraphIR()

    # ========================================================
    # FILE
    # ========================================================

    file_id = stable_id(
        "File",
        filepath,
        filepath
    )

    graph.add_node(
        "File",
        file_id,
        path=filepath,
        name=Path(filepath).name,
        extension=Path(filepath).suffix,
    )

    # ========================================================
    # LOCAL SYMBOL TABLES
    # ========================================================

    definitions = {}
    imports = []
    exports = []

    # ========================================================
    # PASS 1A:
    # DEFINITIONS
    # ========================================================

    def discover_definitions(node):

        # ----------------------------------------------------
        # function Foo() {}
        # ----------------------------------------------------

        if node.type == "function_declaration":

            name_node = node.child_by_field_name("name")

            if name_node:

                name = node_text(
                    name_node,
                    source
                )

                label = (
                    "Component"
                    if is_component_name(name)
                    else "Function"
                )

                definition_id = stable_id(
                    label,
                    filepath,
                    name
                )

                graph.add_node(
                    label,
                    definition_id,
                    name=name,
                    path=filepath,
                    line=line(node),
                    column=column(node),
                    declaration_type="function_declaration",
                )

                graph.add_relationship(
                    file_id,
                    "DEFINES",
                    definition_id,
                    line=line(node),
                )

                definitions[name] = {
                    "id": definition_id,
                    "label": label,
                    "name": name,
                    "line": line(node),
                    "node": node,
                }

        # ----------------------------------------------------
        # const Foo = () => {}
        #
        # const Foo = function () {}
        #
        # const Foo = memo(() => {})
        # ----------------------------------------------------

        if node.type == "variable_declarator":

            name_node = node.child_by_field_name("name")
            value_node = node.child_by_field_name("value")

            if not name_node or not value_node:
                return

            name = node_text(
                name_node,
                source
            )

            if not is_component_name(name):
                return

            is_component = False

            if value_node.type in {
                "arrow_function",
                "function_expression",
            }:

                is_component = True

            # memo(() => ...)
            #
            # forwardRef((props, ref) => ...)
            elif value_node.type == "call_expression":

                function_node = (
                    value_node.child_by_field_name("function")
                )

                if function_node:

                    wrapper_name = node_text(
                        function_node,
                        source
                    )

                    if wrapper_name in {
                        "memo",
                        "forwardRef",
                        "lazy",
                    }:

                        is_component = True

            if not is_component:
                return

            definition_id = stable_id(
                "Component",
                filepath,
                name
            )

            graph.add_node(
                "Component",
                definition_id,
                name=name,
                path=filepath,
                line=line(node),
                column=column(node),
                declaration_type=value_node.type,
            )

            graph.add_relationship(
                file_id,
                "DEFINES",
                definition_id,
                line=line(node),
            )

            definitions[name] = {
                "id": definition_id,
                "label": "Component",
                "name": name,
                "line": line(node),
                "node": node,
            }

        for child in node.named_children:
            discover_definitions(child)

    discover_definitions(
        tree.root_node
    )

    # ========================================================
    # PASS 1B:
    # IMPORTS
    # ========================================================

    def discover_imports(node):

        if node.type == "import_statement":

            statement = node_text(
                node,
                source
            )

            parsed = parse_import_statement(
                statement
            )

            for item in parsed:

                imports.append({
                    **item,
                    "line": line(node),
                })

        for child in node.named_children:
            discover_imports(child)

    discover_imports(
        tree.root_node
    )

    # ========================================================
    # PASS 1C:
    # EXPORTS
    # ========================================================

    exports = parse_exports(
        source.decode(
            "utf-8",
            errors="replace"
        )
    )

    # ========================================================
    # IMPORT NODES
    # ========================================================

    for index, item in enumerate(imports):

        module_name = item["module"]

        import_id = stable_id(
            "Import",
            filepath,
            module_name,
            str(index)
        )

        graph.add_node(
            "Import",
            import_id,
            module=module_name,
            local=item["local"],
            imported=item["imported"],
            kind=item["kind"],
            file=filepath,
            line=item["line"],
        )

        graph.add_relationship(
            file_id,
            "IMPORTS",
            import_id,
            line=item["line"],
        )

    # ========================================================
    # MODULE NODES
    # ========================================================

    module_ids = {}

    for item in imports:

        module_name = item["module"]

        if module_name in module_ids:
            continue

        module_id = stable_id(
            "Module",
            filepath,
            module_name
        )

        module_ids[module_name] = module_id

        graph.add_node(
            "Module",
            module_id,
            name=module_name,
            importer=filepath,
        )

        graph.add_relationship(
            file_id,
            "USES_MODULE",
            module_id
        )

    # ========================================================
    # RENDER / CALL EXTRACTION
    # ========================================================

    def extract_prop_bindings(
        opening_node,
        render_id
    ):

        for child in opening_node.children:

            # ------------------------------------------------
            # normal prop
            #
            # title="Hello"
            # onClick={handleSave}
            # disabled
            # ------------------------------------------------

            if child.type == "jsx_attribute":

                name_node = (
                    child.child_by_field_name("name")
                )

                value_node = (
                    child.child_by_field_name("value")
                )

                if not name_node:
                    continue

                prop_name = node_text(
                    name_node,
                    source
                )

                if value_node:

                    raw_value = node_text(
                        value_node,
                        source
                    )

                    if value_node.type == "string":
                        value_kind = "literal"

                    elif value_node.type == "jsx_expression":
                        value_kind = "expression"

                    else:
                        value_kind = value_node.type

                else:

                    raw_value = "true"
                    value_kind = "boolean"

                prop_id = stable_id(
                    "PropBinding",
                    filepath,
                    prop_name,
                    f"{render_id}|{child.start_byte}"
                )

                graph.add_node(
                    "PropBinding",
                    prop_id,
                    name=prop_name,
                    value=raw_value,
                    value_kind=value_kind,
                    file=filepath,
                    line=line(child),
                    column=column(child),
                    source=node_text(
                        child,
                        source
                    ),
                )

                graph.add_relationship(
                    render_id,
                    "PASSES_PROP",
                    prop_id,
                    line=line(child),
                    name=prop_name,
                    value=raw_value,
                )

            # ------------------------------------------------
            # {...props}
            # ------------------------------------------------

            elif child.type == "jsx_expression":

                expression = node_text(
                    child,
                    source
                ).strip()

                if expression.startswith("{..."):

                    spread_id = stable_id(
                        "PropSpread",
                        filepath,
                        expression,
                        f"{render_id}|{child.start_byte}"
                    )

                    graph.add_node(
                        "PropSpread",
                        spread_id,
                        expression=expression,
                        file=filepath,
                        line=line(child),
                        column=column(child),
                    )

                    graph.add_relationship(
                        render_id,
                        "PASSES_PROP_SPREAD",
                        spread_id,
                        line=line(child),
                        expression=expression,
                    )

    # ========================================================
    # JSX NAME
    # ========================================================

    def jsx_name(opening_node):

        name_node = (
            opening_node.child_by_field_name("name")
        )

        if name_node:
            return node_text(
                name_node,
                source
            )

        return None

    # ========================================================
    # USAGE WALK
    # ========================================================

    def discover_usage(
        node,
        current_scope=None
    ):

        # ----------------------------------------------------
        # FUNCTION DECLARATION SCOPE
        # ----------------------------------------------------

        if node.type == "function_declaration":

            name_node = (
                node.child_by_field_name("name")
            )

            if name_node:

                name = node_text(
                    name_node,
                    source
                )

                if name in definitions:

                    current_scope = definitions[
                        name
                    ]["id"]

        # ----------------------------------------------------
        # ARROW / FUNCTION VARIABLE SCOPE
        # ----------------------------------------------------

        if node.type == "variable_declarator":

            name_node = (
                node.child_by_field_name("name")
            )

            value_node = (
                node.child_by_field_name("value")
            )

            if (
                name_node
                and value_node
            ):

                name = node_text(
                    name_node,
                    source
                )

                if (
                    name in definitions
                    and definitions[name]["label"]
                    == "Component"
                ):

                    current_scope = definitions[
                        name
                    ]["id"]

        # ----------------------------------------------------
        # JSX ELEMENT
        # ----------------------------------------------------

        if node.type in {
            "jsx_element",
            "jsx_self_closing_element",
        }:

            opening = node

            # <Header>...</Header>
            if node.type == "jsx_element":

                opening = next(
                    (
                        child
                        for child in node.named_children
                        if child.type
                        == "jsx_opening_element"
                    ),
                    None
                )

            if opening:

                rendered_name = jsx_name(
                    opening
                )

                if rendered_name:

                    # HTML elements use lowercase.
                    # React components use uppercase.
                    is_component = (
                        rendered_name[0].isupper()
                        or "." in rendered_name
                    )

                    if is_component:

                        render_id = stable_id(
                            "RenderSite",
                            filepath,
                            rendered_name,
                            str(node.start_byte)
                        )

                        graph.add_node(
                            "RenderSite",
                            render_id,
                            name=rendered_name,
                            file=filepath,
                            line=line(node),
                            column=column(node),
                            source=node_text(
                                node,
                                source
                            ),
                        )

                        owner = (
                            current_scope
                            if current_scope
                            else file_id
                        )

                        graph.add_relationship(
                            owner,
                            "RENDERS",
                            render_id,
                            line=line(node),
                        )

                        # ------------------------------------------------
                        # Store unresolved target information.
                        #
                        # We resolve this globally AFTER all files
                        # have been parsed.
                        # ------------------------------------------------

                        graph.nodes[
                            render_id
                        ]["raw_target"] = (
                            rendered_name
                        )

                        extract_prop_bindings(
                            opening,
                            render_id
                        )

        # ----------------------------------------------------
        # FUNCTION CALLS
        # ----------------------------------------------------

        if node.type == "call_expression":

            function_node = (
                node.child_by_field_name("function")
            )

            if function_node:

                called_name = node_text(
                    function_node,
                    source
                )

                if current_scope and called_name:

                    call_id = stable_id(
                        "CallSite",
                        filepath,
                        called_name,
                        str(node.start_byte)
                    )

                    graph.add_node(
                        "CallSite",
                        call_id,
                        name=called_name,
                        file=filepath,
                        line=line(node),
                        source=node_text(
                            node,
                            source
                        ),
                    )

                    graph.add_relationship(
                        current_scope,
                        "CALL_SITE",
                        call_id,
                        line=line(node),
                    )

                    graph.nodes[
                        call_id
                    ]["raw_callee"] = called_name

        # ----------------------------------------------------
        # CHILDREN
        # ----------------------------------------------------

        for child in node.named_children:

            discover_usage(
                child,
                current_scope
            )

    discover_usage(
        tree.root_node
    )

    # ========================================================
    # FILE METADATA
    # ========================================================

    graph.nodes[file_id][
        "definitions"
    ] = list(definitions.keys())

    graph.nodes[file_id][
        "imports"
    ] = imports

    graph.nodes[file_id][
        "exports"
    ] = exports

    return {
        "graph": graph,
        "filepath": filepath,
        "definitions": definitions,
        "imports": imports,
        "exports": exports,
    }


# ============================================================
# IMPORT RESOLUTION
# ============================================================

def resolve_module(
    source_file: str,
    module_name: str,
    target_root: Path
):

    # External npm package:
    #
    # react
    # react-dom
    # lodash
    #
    if not module_name.startswith("."):
        return None

    base = (
        target_root
        / Path(source_file).parent
        / module_name
    ).resolve()

    candidates = [
        base,

        Path(str(base) + ".js"),
        Path(str(base) + ".jsx"),
        Path(str(base) + ".ts"),
        Path(str(base) + ".tsx"),

        base / "index.js",
        base / "index.jsx",
        base / "index.ts",
        base / "index.tsx",
    ]

    for candidate in candidates:

        if candidate.exists() and candidate.is_file():

            return normalize_path(
                str(
                    candidate.relative_to(
                        target_root
                    )
                )
            )

    return None


# ============================================================
# GLOBAL SYMBOL INDEX
# ============================================================

def build_symbol_index(parsed_files):

    by_file_and_name = {}

    default_exports = {}

    for item in parsed_files:

        filepath = item["filepath"]
        definitions = item["definitions"]
        exports = item["exports"]

        for name, definition in definitions.items():

            by_file_and_name[
                (filepath, name)
            ] = definition

        # --------------------------------------------
        # Determine default export
        # --------------------------------------------

        for export in exports:

            if export["exported"] == "default":

                local = export["local"]

                if local in definitions:

                    default_exports[
                        filepath
                    ] = definitions[
                        local
                    ]

    return (
        by_file_and_name,
        default_exports
    )


# ============================================================
# RESOLVE RENDER TARGETS
# ============================================================

def link_render_targets(
    parsed_files,
    all_graph,
    symbol_index,
    default_exports,
    target_root
):

    by_file_and_name = symbol_index

    # --------------------------------------------------------
    # Import lookup
    # --------------------------------------------------------

    imports_by_file = defaultdict(list)

    for item in parsed_files:

        imports_by_file[
            item["filepath"]
        ] = item["imports"]

    # --------------------------------------------------------
    # Process every RenderSite
    # --------------------------------------------------------

    for item in parsed_files:

        filepath = item["filepath"]

        graph = item["graph"]

        local_definitions = item[
            "definitions"
        ]

        imports = imports_by_file[
            filepath
        ]

        import_lookup = {
            imp["local"]: imp
            for imp in imports
            if imp["local"]
        }

        for render_id, node in list(
            graph.nodes.items()
        ):

            if node["label"] != "RenderSite":
                continue

            rendered_name = node.get(
                "raw_target"
            )

            if not rendered_name:
                continue

            # =================================================
            # CASE 1:
            # Local component
            #
            # <Header />
            #
            # where Header is defined in App.jsx
            # =================================================

            if rendered_name in local_definitions:

                target = local_definitions[
                    rendered_name
                ]

                graph.add_relationship(
                    render_id,
                    "TARGETS",
                    target["id"],
                    resolution="local_ast_exact",
                )

                continue

            # =================================================
            # CASE 2:
            # Imported component
            #
            # import Header from "./Header";
            #
            # <Header />
            # =================================================

            if rendered_name in import_lookup:

                imp = import_lookup[
                    rendered_name
                ]

                module_name = imp["module"]

                target_file = resolve_module(
                    filepath,
                    module_name,
                    target_root
                )

                module_id = stable_id(
                    "Module",
                    filepath,
                    module_name
                )

                # Create module -> resolved file.
                if target_file:

                    target_file_id = stable_id(
                        "File",
                        target_file,
                        target_file
                    )

                    if target_file_id not in all_graph.nodes:

                        all_graph.add_node(
                            "File",
                            target_file_id,
                            path=target_file,
                            name=Path(
                                target_file
                            ).name,
                            extension=Path(
                                target_file
                            ).suffix,
                        )

                    all_graph.add_relationship(
                        module_id,
                        "RESOLVES_TO",
                        target_file_id,
                        resolution="relative_import",
                    )

                    # -----------------------------------------
                    # Default import
                    # -----------------------------------------

                    if imp["imported"] == "default":

                        target = default_exports.get(
                            target_file
                        )

                        if target:

                            all_graph.add_relationship(
                                render_id,
                                "TARGETS",
                                target["id"],
                                resolution="default_import",
                            )

                            continue

                    # -----------------------------------------
                    # Named import
                    # -----------------------------------------

                    imported_name = imp[
                        "imported"
                    ]

                    target = by_file_and_name.get(
                        (
                            target_file,
                            imported_name
                        )
                    )

                    if target:

                        all_graph.add_relationship(
                            render_id,
                            "TARGETS",
                            target["id"],
                            resolution="named_import",
                        )

                        continue

                # ------------------------------------------------
                # Import exists, but target can't be resolved.
                #
                # IMPORTANT:
                # do NOT guess.
                # ------------------------------------------------

                unresolved_id = stable_id(
                    "UnresolvedComponent",
                    filepath,
                    rendered_name
                )

                all_graph.add_node(
                    "UnresolvedComponent",
                    unresolved_id,
                    name=rendered_name,
                    source_file=filepath,
                    module=module_name,
                    imported_name=imp["imported"],
                )

                all_graph.add_relationship(
                    render_id,
                    "UNRESOLVED_TARGET",
                    unresolved_id,
                    reason="import_target_not_found",
                )

                continue

            # =================================================
            # CASE 3:
            # Unknown component
            # =================================================

            unresolved_id = stable_id(
                "UnresolvedComponent",
                filepath,
                rendered_name
            )

            all_graph.add_node(
                "UnresolvedComponent",
                unresolved_id,
                name=rendered_name,
                source_file=filepath,
                reason="no_local_or_import_binding",
            )

            all_graph.add_relationship(
                render_id,
                "UNRESOLVED_TARGET",
                unresolved_id,
                reason="symbol_not_resolved",
            )


# ============================================================
# LINK CALL SITES
# ============================================================

def link_calls(
    parsed_files,
    all_graph
):

    for item in parsed_files:

        definitions = item[
            "definitions"
        ]

        graph = item[
            "graph"
        ]

        for call_id, node in list(
            graph.nodes.items()
        ):

            if node["label"] != "CallSite":
                continue

            callee = node.get(
                "raw_callee"
            )

            if not callee:
                continue

            if callee in definitions:

                graph.add_relationship(
                    call_id,
                    "CALLS",
                    definitions[
                        callee
                    ]["id"],
                    resolution="local",
                )


# ============================================================
# BUILD PROJECT
# ============================================================

def build_project(target_directory="data"):

    target_root = Path(
        target_directory
    ).resolve()

    print(
        f"\nScanning: {target_root}"
    )

    parsed_files = []

    # ========================================================
    # PASS 1 — Parse every source file
    # ========================================================

    for root, dirs, files in os.walk(
        target_root
    ):

        dirs[:] = [
            d
            for d in dirs
            if d not in IGNORED_DIRECTORIES
        ]

        for filename in files:

            if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            absolute_file = Path(root) / filename

            relative_file = normalize_path(
                str(
                    absolute_file.relative_to(
                        target_root
                    )
                )
            )

            print(
                f"  Parsing {relative_file}"
            )

            try:

                parsed = parse_file(
                    relative_file,
                    target_root
                )

                parsed_files.append(
                    parsed
                )

            except Exception as exc:

                print(
                    f"  ERROR: {relative_file}"
                )

                print(
                    f"         {exc}"
                )

    # ========================================================
    # GLOBAL INDEX
    # ========================================================

    print(
        "\nBuilding global symbol index..."
    )

    (
        symbol_index,
        default_exports
    ) = build_symbol_index(
        parsed_files
    )

    # ========================================================
    # MERGE GRAPH IR
    # ========================================================

    all_graph = GraphIR()

    for item in parsed_files:

        graph = item[
            "graph"
        ]

        all_graph.nodes.update(
            graph.nodes
        )

        all_graph.relationships.extend(
            graph.relationships
        )

    # ========================================================
    # PASS 2 — Cross-file resolution
    # ========================================================

    print(
        "Resolving imports and component targets..."
    )

    link_render_targets(
        parsed_files,
        all_graph,
        symbol_index,
        default_exports,
        target_root
    )

    # ========================================================
    # REMOVE DUPLICATE RELATIONSHIPS
    # ========================================================

    unique_relationships = {}

    for rel in all_graph.relationships:

        key = (
            rel["source"],
            rel["type"],
            rel["target"],
        )

        unique_relationships[key] = rel

    all_graph.relationships = list(
        unique_relationships.values()
    )

    # ========================================================
    # REBUILD NEO4J
    # ========================================================

    write_to_neo4j(
        all_graph
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    labels = defaultdict(int)

    for node in all_graph.nodes.values():

        labels[
            node["label"]
        ] += 1

    print(
        "\n======================================"
    )

    print(
        "AST GRAPH BUILD COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        f"Files:          {labels['File']}"
    )

    print(
        f"Components:     {labels['Component']}"
    )

    print(
        f"Functions:      {labels['Function']}"
    )

    print(
        f"Render sites:   {labels['RenderSite']}"
    )

    print(
        f"Props:          {labels['PropBinding']}"
    )

    print(
        f"Imports:        {labels['Import']}"
    )

    print(
        f"Modules:        {labels['Module']}"
    )

    print(
        f"Unresolved:     {labels['UnresolvedComponent']}"
    )

    print(
        f"Relationships:  {len(all_graph.relationships)}"
    )


# ============================================================
# NEO4J SCHEMA + WRITE
# ============================================================

def write_to_neo4j(
    graph: GraphIR
):

    with driver.session() as session:

        print(
            "\nClearing previous graph..."
        )

        session.run(
            "MATCH (n) DETACH DELETE n"
        )

        # ----------------------------------------------------
        # Constraints
        # ----------------------------------------------------

        for label in {
            "File",
            "Component",
            "Function",
            "RenderSite",
            "PropBinding",
            "PropSpread",
            "Module",
            "Import",
            "CallSite",
            "UnresolvedComponent",
        }:

            try:

                session.run(
                    f"""
                    CREATE CONSTRAINT
                    IF NOT EXISTS
                    FOR (n:`{label}`)
                    REQUIRE n.id IS UNIQUE
                    """
                )

            except Exception as exc:

                print(
                    f"Schema warning for {label}: {exc}"
                )

        # ----------------------------------------------------
        # Group nodes by label
        # ----------------------------------------------------

        nodes_by_label = defaultdict(list)

        for node in graph.nodes.values():

            properties = {
                key: value
                for key, value in node.items()
                if key != "label"
                and not key.startswith("raw_")
                and key not in {
                    "definitions",
                    "imports",
                    "exports",
                }
            }

            nodes_by_label[
                node["label"]
            ].append(properties)

        # ----------------------------------------------------
        # Insert nodes in batches
        # ----------------------------------------------------

        print(
            "Writing nodes..."
        )

        for label, rows in nodes_by_label.items():

            session.run(
                f"""
                UNWIND $rows AS row
                MERGE (n:`{label}` {{id: row.id}})
                SET n += row
                """,
                rows=rows
            )

        # ----------------------------------------------------
        # Insert relationships in batches
        # ----------------------------------------------------

        relationships_by_type = defaultdict(list)

        for rel in graph.relationships:

            relationships_by_type[
                rel["type"]
            ].append(rel)

        print(
            "Writing relationships..."
        )

        for relation, rows in relationships_by_type.items():

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
                        "properties": row[
                            "properties"
                        ],
                    }
                    for row in rows
                ]
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        driver.verify_connectivity()

        print(
            "Connected to Neo4j."
        )

        build_project(
            "data"
        )

    finally:

        driver.close()