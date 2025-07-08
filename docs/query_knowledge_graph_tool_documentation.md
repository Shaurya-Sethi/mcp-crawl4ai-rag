# Query Knowledge Graph Tool

## Tool Overview

`query_knowledge_graph` is an MCP tool for exploring and querying the Neo4j knowledge graph created by the Crawl4AI RAG server. It allows AI agents and developers to inspect parsed repositories, list classes and methods, and run custom Cypher queries. The tool is primarily used for AI hallucination detection and for understanding repository contents prior to generating or validating code.

This tool is enabled when `USE_KNOWLEDGE_GRAPH=true` and requires a working Neo4j connection. It complements the `parse_github_repository` and `check_ai_script_hallucinations` tools by providing a way to manually browse the data used for validation.

## Prerequisites and Setup

1. Install and run Neo4j (local or cloud). Default URI is `bolt://localhost:7687`.
2. Enable the knowledge graph functionality by setting the following environment variables in your `.env` file:

```bash
USE_KNOWLEDGE_GRAPH=true
NEO4J_URI=bolt://localhost:7687   # or your Neo4j instance
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your password>
```

These variables mirror the configuration shown in the project README【F:README.md†L190-L213】 and in `.env.example`【F:.env.example†L47-L59】.

3. Ensure you have parsed at least one repository using `parse_github_repository`. Without data, most commands will return empty results.

## API Reference

```python
async def query_knowledge_graph(ctx: Context, command: str) -> str
```

- **Parameters**
  - `ctx` (`Context`): MCP server context containing a `repo_extractor` with a Neo4j driver.
  - `command` (`str`): Command string to execute.
- **Returns**: JSON `str` describing the result or error.
- **Errors**: Returns a JSON object with `success: false` and an `error` message if the command fails or if the knowledge graph is disabled.

The function automatically checks that `USE_KNOWLEDGE_GRAPH=true` and that a Neo4j driver is available before executing any query【F:src/crawl4ai_mcp.py†L1308-L1370】.

## Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `repos` | List all repositories in the knowledge graph. **Always run first.** | `repos` |
| `explore <repo>` | Show statistics about a repository (files, classes, functions, methods). | `explore pydantic-ai` |
| `classes` | List up to 20 classes across all repositories. | `classes` |
| `classes <repo>` | List up to 20 classes within a repository. | `classes pydantic-ai` |
| `class <name>` | Show details of a class (methods and attributes). | `class Agent` |
| `method <name>` | Search for methods by name across classes (limit 20). | `method run` |
| `method <name> <class>` | Search for a method within a specific class. | `method __init__ Agent` |
| `function <name>` | Search for standalone functions by name (limit 20). | `function hello` |
| `query <cypher>` | Run a custom Cypher query (results limited to 20). | `query MATCH (c:Class) RETURN c.name` |

## Knowledge Graph Schema

Nodes and relationships are stored as described in the README【F:README.md†L498-L525】 and summarized below:

- **Nodes**
  - `Repository`: `(r:Repository {name})`
  - `File`: `(f:File {path, module_name})`
  - `Class`: `(c:Class {name, full_name})`
  - `Method`: `(m:Method {name, params_list, params_detailed, return_type, args})`
  - `Function`: `(func:Function {name, params_list, params_detailed, return_type, args})`
  - `Attribute`: `(a:Attribute {name, type})`
- **Relationships**
  - `(r:Repository)-[:CONTAINS]->(f:File)`
  - `(f:File)-[:DEFINES]->(c:Class)`
  - `(c:Class)-[:HAS_METHOD]->(m:Method)`
  - `(c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)`
  - `(f:File)-[:DEFINES]->(func:Function)`

## Response Format

Every command returns a JSON string with a common structure:

```json
{
  "success": true,
  "command": "classes",
  "data": { ... },
  "metadata": { ... }
}
```

When an error occurs `success` is `false` and an `error` field is present.

### Examples

**Successful `repos` command**
```json
{
  "success": true,
  "command": "repos",
  "data": {
    "repositories": ["example-repo"]
  },
  "metadata": {
    "total_results": 1,
    "limited": false
  }
}
```

**Error (repository not found)**
```json
{
  "success": false,
  "command": "explore unknown",
  "error": "Repository 'unknown' not found in knowledge graph"
}
```

See `src/crawl4ai_mcp.py` for exact return fields of each handler.

## Error Handling

Possible errors include:
- Knowledge graph disabled (`USE_KNOWLEDGE_GRAPH` not set)【F:src/crawl4ai_mcp.py†L1310-L1318】
- Neo4j connection not available【F:src/crawl4ai_mcp.py†L1319-L1329】
- Empty command or unknown command【F:src/crawl4ai_mcp.py†L1331-L1389】
- Missing arguments for a command (e.g., no repository specified)【F:src/crawl4ai_mcp.py†L1347-L1388】
- Entity not found (`class`, `method`, or `function` not present)【F:src/crawl4ai_mcp.py†L1518-L1561】【F:src/crawl4ai_mcp.py†L1677-L1692】
- Cypher query errors are caught and returned in the response【F:src/crawl4ai_mcp.py†L1710-L1737】

## Integration Examples

### Parsing a Repository and Querying Functions
```python
import asyncio
from crawl4ai_mcp import parse_github_repository, query_knowledge_graph, Context

ctx = Context(fastmcp=None)

# Parse a repository (must end with .git)
result = asyncio.run(parse_github_repository(ctx, "https://github.com/user/repo.git"))
print(result)

# Query for a function after parsing
response = asyncio.run(query_knowledge_graph(ctx, "function hello"))
print(response)
```
This mirrors the approach used in the tests【F:tests/test_query_function.py†L1-L47】.

### Validating Scripts
After exploring the graph, you can validate scripts using `check_ai_script_hallucinations` which uses the same Neo4j data. Example:
```python
from crawl4ai_mcp import check_ai_script_hallucinations
report = asyncio.run(check_ai_script_hallucinations(ctx, script_content="print(1)", filename="example.py"))
```

## Workflow Examples

1. **Explore available repositories**
   ```
   repos
   ```
2. **Inspect a repository**
   ```
   explore my-repo
   classes my-repo
   class SomeClass
   method run SomeClass
   ```
3. **Run a custom query**
   ```
   query MATCH (c:Class)-[:HAS_METHOD]->(m:Method) WHERE m.name="run" RETURN c.full_name
   ```

These steps follow the recommended workflow from the function docstring【F:src/crawl4ai_mcp.py†L1250-L1279】.

## Advanced Usage

- You can send any Cypher query via the `query` command. Results are limited to 20 records to prevent overwhelming responses【F:src/crawl4ai_mcp.py†L1707-L1725】.
- Combine filters to create complex exploration patterns, e.g., find all methods with a given return type or cross-reference repositories.

## Performance Considerations

- Results for `classes`, `method`, `function`, and `query` commands are limited to 20 records by design to keep responses concise.
- Ensure Neo4j is properly indexed for larger datasets to avoid slow queries.
- Parsing large repositories can be time consuming; consider running `parse_github_repository` asynchronously or in batches.

## Related Tools

The knowledge graph functionality consists of three complementary tools:
1. **`parse_github_repository`** – adds repository code to Neo4j for analysis.
2. **`check_ai_script_hallucinations`** – validates Python scripts against the stored graph to detect hallucinated code.
3. **`query_knowledge_graph`** – explored here, used to inspect repositories, classes, methods, and arbitrary Cypher queries.

Together these tools enable comprehensive hallucination detection workflows as described in the README【F:README.md†L498-L538】.
