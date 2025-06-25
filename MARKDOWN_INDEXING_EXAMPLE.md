# Example: Indexing Markdown Files with Crawl4AI RAG MCP Server

Modern documentation sites often use heavy JavaScript that prevents simple web crawling. A convenient workaround is to copy the rendered page as Markdown from your browser and save it locally. The new `index_markdown_file` tool lets you index these files directly so they can be used for retrieval augmented generation (RAG).

## Use Case: OpenAI Documentation
1. Visit [OpenAI's platform docs](https://platform.openai.com/docs).
2. Use your browser's "Copy page as Markdown" option and save the result to `openai_docs.md`.
3. Run the MCP tool to index the file:

```json
{
  "tool": "index_markdown_file",
  "file_path": "openai_docs.md"
}
```

The server will chunk the file, store it in Supabase with contextual embeddings and (if enabled) extract code examples.

## Workflow Steps
1. Save documentation pages as `.md` or `.txt` files.
2. Call `index_markdown_file` for each file. Optionally provide a custom `source_name`.
3. Use `get_available_sources` to see the new source.
4. Query the indexed content with `perform_rag_query` or search code snippets with `search_code_examples`.

## Configuration Tips
- Set `MODEL_CHOICE` in your `.env` file to a lightweight model like `gpt-4o-mini` for summaries and contextual embeddings.
- Enable `USE_CONTEXTUAL_EMBEDDINGS=true` for better retrieval quality.
- Enable `USE_AGENTIC_RAG=true` to store and search code examples from the markdown.

## Benefits Over Web Crawling
- Works offline and avoids issues with JavaScript-heavy sites.
- Lets you curate exactly what content is indexed.
- Fast indexing since no network requests are needed.

Indexing markdown files expands Crawl4AI RAG MCP's flexibility, letting you bring your own documentation or notes into your RAG workflow.
