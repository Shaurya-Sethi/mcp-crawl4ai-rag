"""
Sentinel MCP server for advanced web intelligence.

This server provides tools to crawl websites using Crawl4AI, automatically
detecting the appropriate crawl method based on URL type (sitemap, txt file, or
regular webpage). It also includes repository parsing and validation tools using
Neo4j knowledge graphs.
"""
from mcp.server.fastmcp import FastMCP, Context
from sentence_transformers import CrossEncoder
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
from tempfile import NamedTemporaryFile
from starlette.requests import Request
from starlette.responses import JSONResponse
import requests
import asyncio
import json
import os
import re
import concurrent.futures
import sys

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

# Add knowledge_graphs folder to path for importing knowledge graph modules
knowledge_graphs_path = Path(__file__).resolve().parent.parent / 'knowledge_graphs'
sys.path.append(str(knowledge_graphs_path))

from utils import (
    get_supabase_client, 
    add_documents_to_supabase, 
    search_documents,
    extract_code_blocks,
    generate_code_example_summary,
    add_code_examples_to_supabase,
    update_source_info,
    extract_source_summary,
    search_code_examples,
    LLM_MAX_CONCURRENCY
)

# Import knowledge graph modules
from knowledge_graph_validator import KnowledgeGraphValidator
from parse_repo_into_neo4j import DirectNeo4jExtractor

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Helper functions for Neo4j validation and error handling
def validate_neo4j_connection() -> bool:
    """Check if Neo4j environment variables are configured."""
    return all([
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD")
    ])

def format_neo4j_error(error: Exception) -> str:
    """Format Neo4j connection errors for user-friendly messages."""
    error_str = str(error).lower()
    if "authentication" in error_str or "unauthorized" in error_str:
        return "Neo4j authentication failed. Check NEO4J_USER and NEO4J_PASSWORD."
    elif "connection" in error_str or "refused" in error_str or "timeout" in error_str:
        return "Cannot connect to Neo4j. Check NEO4J_URI and ensure Neo4j is running."
    elif "database" in error_str:
        return "Neo4j database error. Check if the database exists and is accessible."
    else:
        return f"Neo4j error: {str(error)}"

def validate_script_path(script_path: str) -> Dict[str, Any]:
    """Validate script path and return error info if invalid."""
    if not script_path or not isinstance(script_path, str):
        return {"valid": False, "error": "Script path is required"}
    
    if not os.path.exists(script_path):
        return {"valid": False, "error": f"Script not found: {script_path}"}
    
    if not script_path.endswith('.py'):
        return {"valid": False, "error": "Only Python (.py) files are supported"}
    
    try:
        # Check if file is readable
        with open(script_path, 'r', encoding='utf-8') as f:
            f.read(1)  # Read first character to test
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": f"Cannot read script file: {str(e)}"}

def validate_github_url(repo_url: str) -> Dict[str, Any]:
    """Validate GitHub repository URL."""
    if not repo_url or not isinstance(repo_url, str):
        return {"valid": False, "error": "Repository URL is required"}
    
    repo_url = repo_url.strip()
    
    # Basic GitHub URL validation
    if not ("github.com" in repo_url.lower() or repo_url.endswith(".git")):
        return {"valid": False, "error": "Please provide a valid GitHub repository URL"}
    
    # Check URL format
    if not (repo_url.startswith("https://") or repo_url.startswith("git@")):
        return {"valid": False, "error": "Repository URL must start with https:// or git@"}
    
    return {"valid": True, "repo_name": repo_url.split('/')[-1].replace('.git', '')}

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Sentinel MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client
    reranking_model: Optional[CrossEncoder] = None
    knowledge_validator: Optional[Any] = None  # KnowledgeGraphValidator when available
    repo_extractor: Optional[Any] = None       # DirectNeo4jExtractor when available

@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    # Initialize Supabase client
    supabase_client = get_supabase_client()
    
    # Initialize cross-encoder model for reranking if enabled
    reranking_model = None
    if os.getenv("USE_RERANKING", "false") == "true":
        try:
            reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Failed to load reranking model: {e}")
            reranking_model = None
    
    # Initialize Neo4j components if configured and enabled
    knowledge_validator = None
    repo_extractor = None
    
    # Check if knowledge graph functionality is enabled
    knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
    
    if knowledge_graph_enabled:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if neo4j_uri and neo4j_user and neo4j_password:
            try:
                print("Initializing knowledge graph components...")
                
                # Initialize knowledge graph validator
                knowledge_validator = KnowledgeGraphValidator(neo4j_uri, neo4j_user, neo4j_password)
                await knowledge_validator.initialize()
                print("✓ Knowledge graph validator initialized")
                
                # Initialize repository extractor
                repo_extractor = DirectNeo4jExtractor(neo4j_uri, neo4j_user, neo4j_password)
                await repo_extractor.initialize()
                print("✓ Repository extractor initialized")
                
            except Exception as e:
                print(f"Failed to initialize Neo4j components: {format_neo4j_error(e)}")
                knowledge_validator = None
                repo_extractor = None
        else:
            print("Neo4j credentials not configured - knowledge graph tools will be unavailable")
    else:
        print("Knowledge graph functionality disabled - set USE_KNOWLEDGE_GRAPH=true to enable")
    
    context = Crawl4AIContext(
        crawler=crawler,
        supabase_client=supabase_client,
        reranking_model=reranking_model,
        knowledge_validator=knowledge_validator,
        repo_extractor=repo_extractor,
    )
    # expose the lifespan context on the FastMCP instance so HTTP routes can
    # access it without an active MCP request context
    server.lifespan_context = context

    try:
        yield context
    finally:
        # remove reference for cleanup
        server.lifespan_context = None
        # Clean up all components
        await crawler.__aexit__(None, None, None)
        if knowledge_validator:
            try:
                await knowledge_validator.close()
                print("✓ Knowledge graph validator closed")
            except Exception as e:
                print(f"Error closing knowledge validator: {e}")
        if repo_extractor:
            try:
                await repo_extractor.close()
                print("✓ Repository extractor closed")
            except Exception as e:
                print(f"Error closing repository extractor: {e}")

# Initialize FastMCP server
mcp = FastMCP(
    "sentinel-mcp",
    description="Sentinel - Advanced Web Intelligence MCP Server",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051")
)

def rerank_results(model: CrossEncoder, query: str, results: List[Dict[str, Any]], content_key: str = "content") -> List[Dict[str, Any]]:
    """
    Rerank search results using a cross-encoder model.
    
    Args:
        model: The cross-encoder model to use for reranking
        query: The search query
        results: List of search results
        content_key: The key in each result dict that contains the text content
        
    Returns:
        Reranked list of results
    """
    if not model or not results:
        return results
    
    try:
        # Extract content from results
        texts = [result.get(content_key, "") for result in results]
        
        # Create pairs of [query, document] for the cross-encoder
        pairs = [[query, text] for text in texts]
        
        # Get relevance scores from the cross-encoder
        scores = model.predict(pairs)
        
        # Add scores to results and sort by score (descending)
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return reranked
    except Exception as e:
        print(f"Error during reranking: {e}")
        return results

def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.
    
    Args:
        sitemap_url: URL of the sitemap
        
    Returns:
        List of URLs found in the sitemap
    """
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.
    
    Args:
        chunk: Markdown chunk
        
    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def process_code_example(args):
    """
    Process a single code example to generate its summary.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (code, context_before, context_after)
        
    Returns:
        The generated summary
    """
    code, context_before, context_after = args
    return generate_code_example_summary(code, context_before, context_after)


async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())