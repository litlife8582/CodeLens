from neo4j import GraphDatabase

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j+s://b7e8aa7f.databases.neo4j.io"
AUTH = ("b7e8aa7f", "dHNxWigHL4aUi3YZbz6NPcw4oVAEjJaaCc3EZ4nE2uA")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()