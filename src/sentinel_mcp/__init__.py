"""Package exposing the Sentinel MCP server and tools."""
from .server import mcp, Context
from . import tools  # noqa: F401 ensures tool registration via decorators

__all__ = [
    "mcp",
    "Context",
    "tools",
]
