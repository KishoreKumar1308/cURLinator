# Integration Tests

This directory contains integration tests for cURLinator that validate the complete system with real API documentation sites and LLM API calls.

## 📋 Test Files

### 1. `test_documentation_agent_integration.py` (10 tests)

Tests the DocumentationAgent with real API documentation sites.

**Tests:**
- ✅ `test_crawl_with_openapi_spec` - OpenAPI detection and fast path
- ✅ `test_openapi_detection_with_swagger_ui` - Swagger UI detection
- ✅ `test_crawl_without_openapi_spec` - Full crawl path without OpenAPI
- ✅ `test_contextual_enrichment_enabled` - Enrichment is applied when enabled
- ✅ `test_contextual_enrichment_disabled` - Enrichment is not applied when disabled
- ✅ `test_page_classification_accuracy` - Page types are classified correctly
- ✅ `test_max_pages_limit_respected` - max_pages limit is respected
- ✅ `test_headless_mode_works` - Headless browser mode works
- ✅ `test_invalid_url_handling` - Error handling for invalid URLs
- ✅ `test_malformed_url_handling` - Error handling for malformed URLs

**What it validates:**
- Real HTTP requests and web crawling
- OpenAPI spec detection from multiple sources
- Page classification (rule-based)
- Contextual enrichment application
- Configuration parameters (max_depth, max_pages, headless)
- Error handling for network failures

---

### 2. `test_chat_agent_integration.py` (7 tests)

Tests the ChatAgent with real documents and vector database operations.

**Tests:**
- ✅ `test_query_with_documents` - Basic RAG-based query answering
- ✅ `test_query_returns_relevant_sources` - Source citations are relevant
- ✅ `test_curl_command_generation` - cURL commands are generated
- ✅ `test_conversation_history_maintained` - Conversation context is maintained
- ✅ `test_chroma_persistence` - Vector database persists correctly
- ✅ `test_empty_documents_handling` - Error handling for empty documents
- ✅ `test_invalid_query_handling` - Error handling for invalid queries

**What it validates:**
- RAG-based query answering with real LLM
- cURL command generation from natural language
- Conversation history and follow-up questions
- Chroma vector database integration
- Hybrid retrieval (semantic + BM25)
- Source citation accuracy
- Error handling for edge cases

**Requirements:**
- LLM API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY)
- Tests will be skipped if no API key is found

---

### 3. `test_end_to_end_workflow.py` (6 tests)

Tests the complete DocumentationAgent → ChatAgent workflow.

**Tests:**
- ✅ `test_complete_workflow_with_openapi` - Full workflow with OpenAPI spec
- ✅ `test_complete_workflow_without_openapi` - Full workflow without OpenAPI
- ✅ `test_multiple_queries_on_same_documents` - Multiple queries on same docs
- ✅ `test_conversation_context_maintained` - Context maintained across queries
- ✅ `test_document_metadata_preserved` - Metadata preserved through workflow
- ✅ `test_enrichment_improves_retrieval` - Enrichment improves retrieval quality

**What it validates:**
- Complete end-to-end workflow
- Documents from DocumentationAgent work with ChatAgent
- Multiple queries on the same document set
- Conversation context across the workflow
- Metadata preservation
- Impact of contextual enrichment on retrieval

**Requirements:**
- LLM API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY)
- Tests will be skipped if no API key is found

---

## 🚀 Running the Tests

### Run All Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/integration/ --cov=src/curlinator --cov-report=term-missing
```

### Run Specific Test File

```bash
# DocumentationAgent tests
pytest tests/integration/test_documentation_agent_integration.py -v

# ChatAgent tests
pytest tests/integration/test_chat_agent_integration.py -v

# End-to-end workflow tests
pytest tests/integration/test_end_to_end_workflow.py -v
```

### Run Specific Test

```bash
# Run a single test
pytest tests/integration/test_documentation_agent_integration.py::test_crawl_with_openapi_spec -v

# Run with detailed output
pytest tests/integration/test_documentation_agent_integration.py::test_crawl_with_openapi_spec -vv --tb=short
```

### Skip Integration Tests

```bash
# Skip all integration tests (useful for CI)
pytest -m "not integration"

# Skip slow tests
pytest -m "not slow"

# Skip both integration and slow tests
pytest -m "not integration and not slow"
```

---

## ⚙️ Configuration

### Environment Variables

Integration tests use the same configuration as the main application:

```bash
# LLM API Keys (at least one required for ChatAgent tests)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# Optional: LLM Provider (default: gemini)
DEFAULT_LLM_PROVIDER=openai  # or anthropic, gemini
```

### Test Markers

Tests are marked with pytest markers for selective execution:

- `@pytest.mark.integration` - All integration tests
- `@pytest.mark.slow` - Slow tests (end-to-end workflow)

Configure in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "integration: marks tests as integration tests",
    "slow: marks tests as slow tests",
]
```

---

## 📊 Test Coverage

### Current Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| **DocumentationAgent** | 10 | OpenAPI detection, crawling, enrichment, classification |
| **ChatAgent** | 7 | RAG queries, cURL generation, conversation, persistence |
| **End-to-End** | 6 | Complete workflow, metadata preservation, enrichment |
| **Total** | **23** | **Comprehensive integration coverage** |

### What's Tested

✅ **DocumentationAgent:**
- OpenAPI spec detection (Swagger UI, common paths)
- Full web crawling (WholeSiteReader)
- Page classification (rule-based)
- Contextual enrichment (Anthropic's approach)
- Configuration parameters (max_depth, max_pages, headless)
- Error handling (invalid URLs, network failures)

✅ **ChatAgent:**
- RAG-based query answering
- cURL command generation
- Conversation history management
- Chroma vector database integration
- Hybrid retrieval (BM25 + vector search)
- Source citation accuracy
- Error handling (empty docs, invalid queries)

✅ **End-to-End Workflow:**
- DocumentationAgent → ChatAgent integration
- Multiple queries on same document set
- Conversation context maintenance
- Metadata preservation
- Enrichment impact on retrieval

---

## 🐛 Troubleshooting

### "No LLM API key found - skipping test"

**Issue:** ChatAgent and end-to-end tests are skipped.

**Solution:** Add an API key to `.env`:
```bash
OPENAI_API_KEY=sk-...
# OR
ANTHROPIC_API_KEY=sk-ant-...
# OR
GEMINI_API_KEY=...
```

### "Browser not found" Error

**Issue:** Playwright browser not installed.

**Solution:**
```bash
uv run playwright install chromium
```

### Tests are Slow

**Issue:** Integration tests make real HTTP requests and LLM API calls.

**Expected behavior:** Integration tests are intentionally slow because they:
- Crawl real API documentation sites
- Make real LLM API calls
- Build vector databases
- Perform hybrid retrieval

**Typical times:**
- DocumentationAgent tests: 10-30 seconds each
- ChatAgent tests: 5-15 seconds each
- End-to-end tests: 30-60 seconds each

**To speed up:**
- Run specific tests instead of all tests
- Use `-m "not slow"` to skip slow tests
- Run in parallel with `pytest-xdist` (not recommended for integration tests)

### Network Errors

**Issue:** Tests fail with network errors.

**Possible causes:**
1. No internet connection
2. API documentation site is down
3. Rate limiting

**Solution:**
- Check internet connection
- Try again later
- Use different test URLs

---

## 📈 Performance Expectations

| Test Type | Count | Avg Time | Total Time |
|-----------|-------|----------|------------|
| **DocumentationAgent** | 10 | 15s | ~2.5 min |
| **ChatAgent** | 7 | 10s | ~1.2 min |
| **End-to-End** | 6 | 45s | ~4.5 min |
| **Total** | **23** | **20s** | **~8 min** |

**Note:** Times vary based on:
- Network speed
- LLM API response time
- Site complexity
- System resources

---

## 🎯 Best Practices

### When to Run Integration Tests

✅ **Run integration tests:**
- Before committing major changes
- Before creating a pull request
- After refactoring agents
- When adding new features
- Before releases

❌ **Don't run integration tests:**
- On every file save (too slow)
- In pre-commit hooks (too slow)
- For quick iteration (use unit tests)

### CI/CD Integration

For CI/CD pipelines, consider:

```yaml
# Example GitHub Actions workflow
- name: Run unit tests
  run: pytest tests/unit/ -v

- name: Run integration tests
  run: pytest tests/integration/ -v
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  # Only run on main branch or PRs
  if: github.ref == 'refs/heads/main' || github.event_name == 'pull_request'
```

---

## 📚 Additional Resources

- **Unit Tests:** `tests/unit/` - Fast, isolated tests for utilities
- **Agent Tests:** `tests/agents/` - Unit tests for agent logic
- **Examples:** `examples/` - Working examples of agent usage
- **Documentation:** `examples/README.md` - Guide to using the agents

---

**Last Updated:** 2025-11-14

