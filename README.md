# cURLinator

AI-powered API documentation crawler and query engine.

## Features

- **DocumentationAgent**: Crawls and processes API documentation sites

  - OpenAPI/Swagger spec detection
  - Intelligent page classification
  - Contextual enrichment for better RAG retrieval
  - Batch crawling support

- **ChatAgent**: RAG-based querying and cURL command generation
  - Hybrid retrieval (BM25 + vector search)
  - Conversation history management
  - Persistent vector storage with Chroma
  - Automatic cURL command generation

## Installation

```bash
# Clone the repository
git clone https://github.com/kishorekumar1308/curlinator.git
cd curlinator/curlinator

# Install in editable mode
pip install -e .

# For development (includes testing and linting tools)
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from curlinator.agents import DocumentationAgent, ChatAgent

# Crawl API documentation
doc_agent = DocumentationAgent(headless=True, max_pages=50)
documents = await doc_agent.execute("https://docs.stripe.com/api")

# Create chat agent and index documents
chat_agent = ChatAgent(
    documents=documents,
    collection_name="stripe_api",
)

# Query the documentation
result = await chat_agent.execute("How do I create a payment intent?")
print(result['curl_command'])
print(result['explanation'])
```

### Batch Crawling

```python
from curlinator.agents import DocumentationAgent

# Initialize agent
doc_agent = DocumentationAgent(max_pages=100, max_depth=3)

# Initialize crawl state
crawl_state = doc_agent.initialize_crawl_state("https://docs.example.com")

# Crawl in batches
batch_size = 10
all_documents = []

while True:
    batch_docs, crawl_state, is_complete = await doc_agent.execute_batch(
        base_url="https://docs.example.com",
        batch_size=batch_size,
        crawl_state=crawl_state,
    )

    all_documents.extend(batch_docs)

    if is_complete:
        break

# Cleanup
doc_agent.cleanup_crawl_state(crawl_state)
```

### Custom LLM Configuration

```python
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from curlinator.agents import DocumentationAgent, ChatAgent

# Use custom LLM
llm = OpenAI(model="gpt-4", api_key="your-api-key")
embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key="your-api-key")

# Create agents with custom LLM
doc_agent = DocumentationAgent(llm=llm)
chat_agent = ChatAgent(llm=llm, embed_model=embed_model)
```

## Configuration

Set environment variables in `.env`:

```bash
# LLM Provider (choose one)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# Vector Database
VECTOR_DB_PATH=./data/vector_db

# Crawler Settings
CRAWLER_MAX_PAGES=50
CRAWLER_TIMEOUT=30000
```

## Development

```bash
# Run tests
pytest tests/

# Run linting
ruff check src/ tests/

# Run formatting
ruff format src/ tests/

# Build package
python -m build
```

## Acknowledgements

This project is built with the following open-source technologies:

- [**Python**](https://www.python.org/) - Core programming language
- [**LlamaIndex**](https://www.llamaindex.ai/) - RAG framework for document indexing and retrieval
- [**Chroma**](https://www.trychroma.com/) - Vector database for semantic search
- [**Selenium**](https://www.selenium.dev/) - Web scraping and browser automation
- [**BeautifulSoup4**](https://www.crummy.com/software/BeautifulSoup/) - HTML parsing and content extraction
- [**OpenAI**](https://openai.com/) / [**Anthropic**](https://www.anthropic.com/) / [**Google Gemini**](https://ai.google.dev/) - LLM providers for intelligent query processing
