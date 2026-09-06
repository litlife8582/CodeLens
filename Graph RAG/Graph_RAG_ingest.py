import os
import argparse
from dotenv import load_dotenv
from neo4j import GraphDatabase
import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser

# 1. Load Environment Variables & Connect to Neo4j
load_dotenv(override=True)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_URI or not NEO4J_PASSWORD:
    raise ValueError("Missing NEO4J_URI or NEO4J_PASSWORD in .env file.")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

# 2. Initialize Tree-sitter Parser for JS / JSX
JS_LANGUAGE = Language(tsjs.language())
ts_parser = Parser(JS_LANGUAGE)


# 3. Deterministic AST Extraction Logic
def extract_ast_elements(code_bytes, filename):
    tree = ts_parser.parse(code_bytes)
    edges = []
    nodes = set()

    nodes.add(("File", filename))

    def traverse_ast(node, current_func=None):
        if node.type == "import_statement":
            source = node.child_by_field_name("source")
            if source:
                module_name = source.text.decode("utf-8").strip("'\"")
                nodes.add(("Module", module_name))
                edges.append((("File", filename), "IMPORTS", ("Module", module_name)))

        elif node.type in ("function_declaration", "arrow_function", "function"):
            func_name_node = node.child_by_field_name("name")
            func_name = func_name_node.text.decode("utf-8") if func_name_node else "anonymous"

            node_label = "Component" if func_name[0].isupper() else "Function"
            nodes.add((node_label, func_name))
            edges.append((("File", filename), "DEFINES", (node_label, func_name)))

            previous_func = current_func
            current_func = (node_label, func_name)

            for child in node.children:
                traverse_ast(child, current_func)

            current_func = previous_func
            return

        elif node.type == "jsx_opening_element" or node.type == "jsx_self_closing_element":
            name_node = node.child_by_field_name("name")
            if name_node:
                rendered_elem = name_node.text.decode("utf-8")
                if rendered_elem[0].isupper():
                    nodes.add(("Component", rendered_elem))
                    caller = current_func if current_func else ("File", filename)
                    edges.append((caller, "RENDERS", ("Component", rendered_elem)))

        elif node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node:
                callee_name = func_node.text.decode("utf-8")
                if current_func and callee_name != current_func[1]:
                    nodes.add(("Function", callee_name))
                    edges.append((current_func, "CALLS", ("Function", callee_name)))

        for child in node.children:
            traverse_ast(child, current_func)

    traverse_ast(tree.root_node)
    return nodes, edges


# 4. Idempotency helpers — this is the key architectural change.
def graph_is_populated() -> bool:
    """Check whether the KG already has data, so we don't silently rebuild it."""
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) AS node_count")
        return result.single()["node_count"] > 0


def clear_graph():
    """Wipe the existing graph. Only called when --force is passed."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


# 5. Ingest All Codebase Files into Neo4j
def build_ast_graph(target_directory="data"):
    print(f"\nScanning directory: '{target_directory}' for codebase files...")

    total_files = 0
    all_nodes = set()
    all_edges = []

    for root, _, files in os.walk(target_directory):
        for file in files:
            if file.endswith((".js", ".jsx", ".ts", ".tsx")):
                total_files += 1
                filepath = os.path.join(root, file)

                with open(filepath, "rb") as f:
                    code_bytes = f.read()

                nodes, edges = extract_ast_elements(code_bytes, file)
                all_nodes.update(nodes)
                all_edges.extend(edges)

    if total_files == 0:
        print(f"Warning: No valid files found inside '{target_directory}'.")
        return

    print(f"Parsed {total_files} files. Writing AST structures to Neo4j...")
    with driver.session() as session:
        for label, name in all_nodes:
            query = f"MERGE (n:`{label}` {{name: $name}})"
            session.run(query, name=name)

        for (src_label, src_name), rel, (dest_label, dest_name) in all_edges:
            query = f"""
            MATCH (a:`{src_label}` {{name: $src_name}})
            MATCH (b:`{dest_label}` {{name: $dest_name}})
            MERGE (a)-[r:`{rel}`]->(b)
            """
            session.run(query, src_name=src_name, dest_name=dest_name)

    print("Graph build complete.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Build the AST knowledge graph once. Re-run only with --force."
    )
    arg_parser.add_argument("--target-dir", default="data", help="Directory of source files to scan")
    arg_parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe and rebuild the graph even if it already contains data",
    )
    args = arg_parser.parse_args()

    try:
        driver.verify_connectivity()
        print("Connected to Neo4j successfully.")

        if graph_is_populated():
            if args.force:
                print("Existing graph detected. --force passed: clearing and rebuilding.")
                clear_graph()
            else:
                print(
                    "The knowledge graph already contains data.\n"
                    "Skipping build so this script doesn't get re-run by accident.\n"
                    "Pass --force if you really want to wipe and rebuild it.\n"
                    "Otherwise, go straight to rag_pipeline.py / evaluate_hallucination.py "
                    "— they read from the graph that's already there."
                )
                raise SystemExit(0)

        build_ast_graph(args.target_dir)
        print(
            "\nKG build finished. You can now query and evaluate it as many times as "
            "you want via rag_pipeline.py / evaluate_hallucination.py without "
            "re-running this script."
        )

    finally:
        driver.close()