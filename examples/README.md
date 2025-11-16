# cURLinator Examples

This directory contains example scripts demonstrating how to use cURLinator's agents.

## 📋 Available Examples

### 1. `my_test.py` - Quick Start Example

**Purpose:** Minimal working example to get started quickly.

**What it demonstrates:**
- Basic DocumentationAgent usage
- How to configure the agent (max_depth, max_pages, enrichment)
- Inspecting the returned `List[Document]`
- Accessing document metadata
- Detecting OpenAPI specs vs full crawls
- Saving results to JSON

**Usage:**
```bash
uv run python examples/my_test.py
```

**Configuration:**
Edit the script to change:
- `api_url` - The API documentation URL to crawl
- `max_depth` - Maximum crawl depth (default: 2)
- `max_pages` - Maximum pages to crawl (default: 20)
- `enable_enrichment` - Add contextual prefixes (default: True)
- `verbose` - Show detailed logging (default: True)

**Expected Output:**
- Document count and types
- OpenAPI detection status
- Sample documents with metadata
- Enrichment statistics
- Saved JSON file with all documents

---

### 2. `test_documentation_agent.py` - Comprehensive Test Suite

**Purpose:** Interactive test suite with multiple API examples.

**What it demonstrates:**
- Testing with different API types:
  - JSONPlaceholder (no OpenAPI spec - full crawl)
  - Swagger Petstore (has OpenAPI spec - fast path)
  - Custom URL (your choice)
- Helper function for displaying results
- Comparing OpenAPI vs crawl approaches
- Detailed metadata inspection

**Usage:**
```bash
uv run python examples/test_documentation_agent.py
```

**Interactive Menu:**
```
Choose test:
1. JSONPlaceholder (Simple API, no OpenAPI spec)
2. Swagger Petstore (Has OpenAPI 3.0)
3. Custom URL (Enter your own API documentation URL)
4. Run all tests (JSONPlaceholder + Petstore)
```

**Expected Output:**
- Document summary with type breakdown
- OpenAPI detection results
- Sample documents with full metadata
- Saved JSON files for each test

---

### 3. `test_end_to_end_workflow.py` - Complete Workflow

**Purpose:** Full end-to-end workflow: DocumentationAgent → ChatAgent.

**What it demonstrates:**
- Complete cURLinator workflow
- Crawling documentation with DocumentationAgent
- Indexing documents in Chroma vector database
- Querying with ChatAgent to generate cURL commands
- Conversation history with follow-up questions
- Hybrid retrieval (semantic + BM25)

**Usage:**
```bash
uv run python examples/test_end_to_end_workflow.py
```

**Requirements:**
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env` file
- Internet connection for crawling
- First run will download local embedding model (~90MB)

**Configuration:**
Edit the script to change:
- `BASE_URL` - API documentation site to crawl
- `LLM_PROVIDER` - "openai" or "anthropic"
- `ENABLE_ENRICHMENT` - Enable contextual enrichment
- `USE_LLM_CLASSIFICATION` - Use LLM for page classification

**Expected Output:**
- Crawl progress and statistics
- Vector database indexing
- Interactive chat queries
- Generated cURL commands
- Source citations

---

## 🚀 Quick Start

### Prerequisites

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Set up API keys (optional for DocumentationAgent, required for ChatAgent):**
   ```bash
   cp .env.example .env
   # Edit .env and add your API key:
   # OPENAI_API_KEY=sk-...
   # OR
   # ANTHROPIC_API_KEY=sk-ant-...
   ```

### Run Your First Example

```bash
# Start with the simplest example
uv run python examples/my_test.py
```

This will:
1. Crawl https://petstore3.swagger.io
2. Detect the OpenAPI spec (fast path)
3. Parse ~10-15 documents
4. Save results to `my_test_results.json`

---

## 📊 Understanding the Output

### DocumentationAgent Returns `List[Document]`

Each `Document` has:
- **`text`**: The document content (with optional contextual enrichment)
- **`metadata`**: Dictionary with information about the document

### Common Metadata Fields

| Field | Description | Example |
|-------|-------------|---------|
| `url` | Source URL | `"https://api.example.com/users"` |
| `title` | Page title | `"Create User"` |
| `page_type` | Type of page | `"api_endpoint"`, `"guide"`, `"tutorial"` |
| `source` | How it was obtained | `"openapi"`, `"crawl"` |
| `type` | Document type (OpenAPI) | `"api_endpoint"`, `"api_overview"` |
| `method` | HTTP method (endpoints) | `"POST"`, `"GET"` |
| `endpoint` | API endpoint path | `"/users"` |
| `tags` | OpenAPI tags | `"users, authentication"` |
| `contextually_enriched` | Was enriched? | `true`, `false` |

### Page Types

The agent classifies pages into these types:
- `api_endpoint` - Individual API endpoint documentation
- `api_overview` - API overview/introduction
- `api_reference` - API reference documentation
- `authentication` - Authentication/authorization docs
- `guide` - How-to guides
- `tutorial` - Step-by-step tutorials
- `quickstart` - Getting started guides
- `sdk` - SDK documentation
- `webhook` - Webhook documentation
- `error` - Error handling documentation
- `changelog` - API changelog
- `unknown` - Unclassified pages

---

## 🎯 Common Use Cases

### 1. Test OpenAPI Detection

```bash
# Test with an API that has OpenAPI spec
uv run python examples/my_test.py
# Edit api_url to: "https://petstore3.swagger.io"
```

**Expected:** Fast path, ~10-15 documents from OpenAPI spec

### 2. Test Full Crawl

```bash
# Test with an API without OpenAPI spec
uv run python examples/my_test.py
# Edit api_url to: "https://jsonplaceholder.typicode.com"
```

**Expected:** Full crawl, documents from HTML pages

### 3. Test Large API

```bash
# Test with a real-world API
uv run python examples/my_test.py
# Edit api_url to: "https://stripe.com/docs/api"
# Increase max_pages to 50
```

**Expected:** OpenAPI spec detected, many endpoint documents

### 4. Compare Approaches

```bash
# Run the comprehensive test suite
uv run python examples/test_documentation_agent.py
# Choose option 4 to run all tests
```

**Expected:** See difference between OpenAPI and crawl approaches

---

## 🔧 Troubleshooting

### "No API key found" Warning

**Issue:** You see a warning about missing API keys.

**Solution:** This is OK for DocumentationAgent! API keys are only needed for:
- LLM-based page classification (optional, rule-based works fine)
- ChatAgent queries (required)

To enable LLM features, add to `.env`:
```bash
OPENAI_API_KEY=sk-...
# OR
ANTHROPIC_API_KEY=sk-ant-...
```

### "Crawled 0 documents"

**Issue:** Agent returns empty list.

**Possible causes:**
1. Invalid URL
2. Site blocks crawlers
3. Network error

**Solution:**
- Check the URL is correct
- Try with `verbose=True` to see detailed logs
- Test with known-working URLs (Petstore, JSONPlaceholder)

### "Browser not found" Error

**Issue:** Playwright browser not installed.

**Solution:**
```bash
uv run playwright install chromium
```

### Slow Crawling

**Issue:** Crawling takes a long time.

**Solution:**
- Reduce `max_pages` (e.g., 10-20 for testing)
- Reduce `max_depth` (e.g., 2 for testing)
- Use `headless=True` (default, faster)

---

## 📚 Next Steps

After running the examples:

1. **Inspect the JSON output** to understand document structure
2. **Try different API documentation sites** to test robustness
3. **Use documents with ChatAgent** for RAG-based queries
4. **Adjust configuration** to optimize for your use case

### Using Documents with ChatAgent

```python
from curlinator.agents import DocumentationAgent, ChatAgent

# Step 1: Crawl documentation
doc_agent = DocumentationAgent(verbose=True)
documents = await doc_agent.execute("https://api.example.com")

# Step 2: Query with ChatAgent
chat_agent = ChatAgent()
response = await chat_agent.query(
    documents=documents,
    query="How do I create a new user?"
)

print(response["response"])
print(response["curl_command"])
```

See `test_end_to_end_workflow.py` for a complete example.

---

## 🤝 Contributing

Found an issue or want to add an example? Please open an issue or PR!

---

**Last Updated:** 2025-11-14

