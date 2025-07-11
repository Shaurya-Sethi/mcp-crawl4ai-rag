# Sentinel MCP Server

Sentinel is an advanced implementation of the [Model Context Protocol](https://modelcontextprotocol.io) for building web intelligence agents. It combines powerful crawling from [Crawl4AI](https://crawl4ai.com) with modern retrieval augmented generation (RAG) techniques and optional Neo4j knowledge graphs.

## Features
- **Web Crawling** through Crawl4AI with configurable browser options
- **Vector Storage** using Supabase or your own Postgres/pgvector instance
- **RAG Enhancements** including contextual embeddings, hybrid search, agentic RAG, and cross-encoder reranking
- **Knowledge Graph** utilities for validating repositories and exploring code structures

## Installation
```bash
git clone https://github.com/Shaurya-Sethi/sentinel-mcp.git
cd sentinel-mcp
python -m venv .venv && source .venv/bin/activate
pip install -e .
crawl4ai-setup
```

## Configuration
Copy `.env.example` to `.env` and fill in the required values. Key settings include:
- `HOST` / `PORT` – network configuration when using SSE transport
- `OPENAI_API_KEY` – API key for embeddings and summaries
- `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` – for vector storage
- RAG toggles such as `USE_CONTEXTUAL_EMBEDDINGS` or `USE_KNOWLEDGE_GRAPH`

## Running
### Local
```bash
python -m sentinel_mcp.server
```

### Docker
Build the image:
```bash
docker build -t ghcr.io/shaurya-sethi/sentinel-mcp:latest .
```
Run it:
```bash
docker run --env-file .env -p 8051:8051 ghcr.io/shaurya-sethi/sentinel-mcp:latest
```

## Developer Guide
Implement new tools in `src/sentinel_mcp/tools.py` using the `@mcp.tool()` decorator. Import the `mcp` instance from `sentinel_mcp.server` and document each tool's parameters and output clearly.

Sentinel is designed as a foundation for creating powerful web intelligence agents and RAG pipelines. Feel free to extend it for your own projects.
