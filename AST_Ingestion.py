import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser

# 1. Load Environment Variables & Connect to Neo4j
load_dotenv()

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
parser = Parser(JS_LANGUAGE)

# 3. Deterministic AST Extraction Logic
def extract_ast_elements(code_bytes, filename):
    tree = parser.parse(code_bytes)
    edges = []
    nodes = set()

    # Always register the file as a foundational node
    nodes.add(("File", filename))

    # FIX: We removed 'nonlocal current_func' since it is passed as a parameter
    def traverse_ast(node, current_func=None):
        
        # A. Detect Import Statements: import X from './Module'
        if node.type == "import_statement":
            source = node.child_by_field_name("source")
            if source:
                module_name = source.text.decode("utf-8").strip("'\"")
                nodes.add(("Module", module_name))
                edges.append((("File", filename), "IMPORTS", ("Module", module_name)))

        # B. Detect Function / Component Declarations
        elif node.type in ("function_declaration", "arrow_function", "function"):
            func_name_node = node.child_by_field_name("name")
            func_name = func_name_node.text.decode("utf-8") if func_name_node else "anonymous"
            
            # If function name starts with uppercase, classify as React Component
            node_label = "Component" if func_name[0].isupper() else "Function"
            nodes.add((node_label, func_name))
            edges.append((("File", filename), "DEFINES", (node_label, func_name)))
            
            # Update the current function scope for children nodes
            current_func = (node_label, func_name)
            
            # Recursively traverse inside the function body
            for child in node.children:
                traverse_ast(child, current_func)
                
            return # Prevent double traversal

        # C. Detect JSX Elements / Component Renders: <Header />
        elif node.type == "jsx_opening_element" or node.type == "jsx_self_closing_element":
            name_node = node.child_by_field_name("name")
            if name_node:
                rendered_elem = name_node.text.decode("utf-8")
                # Only track custom React components (starts with uppercase)
                if rendered_elem[0].isupper():
                    nodes.add(("Component", rendered_elem))
                    caller = current_func if current_func else ("File", filename)
                    edges.append((caller, "RENDERS", ("Component", rendered_elem)))

        # D. Detect Standard Function Calls: helperFunc()
        elif node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node:
                callee_name = func_node.text.decode("utf-8")
                if current_func and callee_name != current_func[1]:
                    nodes.add(("Function", callee_name))
                    edges.append((current_func, "CALLS", ("Function", callee_name)))

        # Continue traversing all child nodes passing the current function context down
        for child in node.children:
            traverse_ast(child, current_func)

    traverse_ast(tree.root_node)
    return nodes, edges

# 4. Ingest All Codebase Files into Neo4j
def build_ast_graph(target_directory="data"):
    print(f"\nScanning directory: '{target_directory}' for codebase files...")
    
    total_files = 0
    all_nodes = set()
    all_edges = []

    # Recursive directory walk to scan subdirectories (e.g., data/src/)
    for root, _, files in os.walk(target_directory):
        for file in files:
            if file.endswith((".js", ".jsx", ".ts", ".tsx")):
                total_files += 1
                filepath = os.path.join(root, file)
                
                with open(filepath, "rb") as f:
                    code_bytes = f.read()

                # Extract AST nodes and edges
                nodes, edges = extract_ast_elements(code_bytes, file)
                all_nodes.update(nodes)
                all_edges.extend(edges)

    print(f"Found {total_files} code file(s).")
    print(f"Parsed {len(all_nodes)} unique nodes and {len(all_edges)} structural edges.\n")

    if total_files == 0:
        print(f"Warning: No .js/.jsx/.ts/.tsx files found inside '{target_directory}'.")
        return

    # Push to Neo4j
    print("Writing AST structures to Neo4j...")
    with driver.session() as session:
        # Create Nodes
        for label, name in all_nodes:
            query = f"MERGE (n:`{label}` {{name: $name}})"
            session.run(query, name=name)

        # Create Relationships (Edges)
        for (src_label, src_name), rel, (dest_label, dest_name) in all_edges:
            print(f"  [+] ({src_label}: {src_name}) -[:{rel}]-> ({dest_label}: {dest_name})")
            query = f"""
            MATCH (a:`{src_label}` {{name: $src_name}})
            MATCH (b:`{dest_label}` {{name: $dest_name}})
            MERGE (a)-[r:`{rel}`]->(b)
            """
            session.run(query, src_name=src_name, dest_name=dest_name)

    print("\nDeterministic AST Graph successfully ingested into Neo4j!")

if __name__ == "__main__":
    try:
        # Ensure driver connectivity
        driver.verify_connectivity()
        print("Connected to Neo4j AuraDB successfully.")
        
        # Build the graph from the data folder
        build_ast_graph("data")
    finally:
        driver.close()