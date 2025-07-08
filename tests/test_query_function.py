import asyncio
import os
import subprocess
import textwrap
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import pytest

from knowledge_graphs.parse_repo_into_neo4j import DirectNeo4jExtractor
from src.crawl4ai_mcp import query_knowledge_graph, Context

class DummyRC:
    def __init__(self, extractor):
        self.lifespan_context = type("lc", (), {"repo_extractor": extractor})()

class DummyContext(Context):
    def __init__(self, extractor):
        super().__init__(fastmcp=None)
        self.request_context = DummyRC(extractor)

@pytest.fixture(scope="session")
async def extractor(tmp_path_factory):
    neo4j_dir = tmp_path_factory.mktemp("neo4j")
    # start Neo4j in test harness if available
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "test")

    extr = DirectNeo4jExtractor(uri, user, password)
    await extr.initialize()
    yield extr
    await extr.close()

@pytest.fixture(scope="session")
async def simple_repo(tmp_path_factory, extractor):
    repo_dir = tmp_path_factory.mktemp("repo")
    (repo_dir / "mod.py").write_text("""\

def hello(x, y):
    return x + y
""")
    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "you@example.com"], cwd=repo_dir)
    subprocess.run(["git", "config", "user.name", "Your Name"], cwd=repo_dir)
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)

    await extractor.analyze_repository(str(repo_dir))
    return repo_dir

@pytest.mark.asyncio
async def test_query_function(extractor, simple_repo):
    ctx = DummyContext(extractor)
    result_json = await query_knowledge_graph(ctx, "function hello")
    assert 'hello' in result_json
