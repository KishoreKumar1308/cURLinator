# cURLinator

**AI-powered API testing and exploration platform**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-502%20passing-brightgreen.svg)](tests/)

---

## 🎯 What is cURLinator?

cURLinator eliminates friction in API integration by providing a conversational interface to discover, test, and execute API calls against any API documentation.

**Key capabilities:**

- 🔍 **Scrapes and understands** API documentation from any website
- 💬 **Answers questions** about APIs using natural language
- ⚡ **Generates working cURL commands** from conversational queries
- 🧠 **Maintains context** across multiple questions for iterative exploration

---

## ✨ Key Features

- **Smart Documentation Crawling**: Auto-detects OpenAPI/Swagger specs or crawls full sites
- **RAG-Powered Queries**: Hybrid retrieval (semantic + BM25) for accurate answers
- **Conversation History**: Server-side session management with automatic summarization
- **Multi-LLM Support**: Works with OpenAI, Anthropic, Google Gemini, or local models
- **Collection Sharing**: Share API documentation collections with team members

---

## 📦 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/kishorekumar1308/curlinator.git
cd curlinator
uv sync  # or: pip install -e .
```

### 2. Configure Environment

Create a `.env` file:

```bash
# Database (required)
DATABASE_URL=postgresql://user:password@localhost:5432/curlinator_dev

# LLM Provider (at least one required)
GEMINI_API_KEY=your-key-here
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Optional: Encryption key for API keys (generate with: openssl rand -base64 32)
API_KEY_ENCRYPTION_KEY=your-encryption-key-here
```

### 3. Set Up Database

```bash
# Run migrations
alembic upgrade head

# Create admin user
python scripts/create_admin.py
```

### 4. Start the API Server

```bash
uvicorn curlinator.api.main:app --reload
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

---

## 💡 Usage Examples

### Using the REST API

```bash
# 1. Register a user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"SecurePass123"}'

# 2. Crawl API documentation
curl -X POST http://localhost:8000/api/v1/crawl \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://petstore3.swagger.io",
    "max_pages": 50,
    "embedding_provider": "LOCAL"
  }'

# 3. Query the documentation
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "crawl_abc123...",
    "message": "How do I create a new pet?"
  }'
```

### Using Python SDK

```python
from curlinator.agents import DocumentationAgent, ChatAgent

# Crawl documentation
doc_agent = DocumentationAgent(headless=True)
documents = await doc_agent.execute("https://petstore3.swagger.io")

# Query with chat agent
chat_agent = ChatAgent(collection_name="petstore_api")
await chat_agent.index_documents(documents)
result = await chat_agent.execute("How do I create a new pet?")
print(result['curl_command'])
```

---

## 📚 API Documentation

For detailed REST API documentation including all endpoints, request/response examples, and error codes, see:

**[📖 Complete API Documentation](API_SETUP.md)**

### Quick API Reference

- **Authentication**: `/api/v1/auth/register`, `/api/v1/auth/login`
- **Crawling**: `/api/v1/crawl`
- **Chat**: `/api/v1/chat`
- **Collections**: `/api/v1/collections/*`
- **Sessions**: `/api/v1/sessions/*`
- **Sharing**: `/api/v1/collections/{name}/share`
- **Health**: `/health`

Interactive API documentation is available at `http://localhost:8000/docs` when running the server.

---

## 🧪 Testing

cURLinator has comprehensive test coverage with **502 passing tests**.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/curlinator --cov-report=html
```

---

## 📊 Monitoring

cURLinator includes production-ready monitoring and observability:

- **Sentry Error Tracking** - Real-time error monitoring and performance tracking
- **Health Checks** (`/health`) - Database, Chroma, system resources, LLM API connectivity
- **Structured Logging** - Correlation IDs, request tracing, audit logs

### Quick Setup

```bash
# Sentry (optional)
SENTRY_DSN=https://your-dsn@sentry.io/project-id

# Prometheus scraping
curl http://localhost:8000/metrics
```

---

## 💾 Production Deployment

### Database Resilience

cURLinator includes automatic database connection retry with exponential backoff for cloud deployments:

```bash
DATABASE_MAX_RETRIES=5  # Default: 5 attempts
```

### Backup & Recovery

**Quick Backup:**

```bash
pg_dump -h localhost -U curlinator -d curlinator_db -F c -f backup.dump
```

**Quick Restore:**

```bash
pg_restore -d curlinator_db backup.dump
```

For detailed backup strategies, disaster recovery procedures, and cloud deployment guides, see the [API Documentation](docs/API_README.md#database-backup).

---

## 🛠️ Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
curlinator/
├── src/curlinator/
│   ├── agents/              # DocumentationAgent, ChatAgent
│   ├── api/                 # FastAPI application
│   ├── utils/               # Utilities and helpers
│   └── config/              # Configuration
├── tests/                   # Test suite (502 tests)
├── alembic/                 # Database migrations
└── scripts/                 # Utility scripts
```

## 🙏 Acknowledgments

Built with [LlamaIndex](https://www.llamaindex.ai/), [Chroma](https://www.trychroma.com/), [FastAPI](https://fastapi.tiangolo.com/), and [PostgreSQL](https://www.postgresql.org/).
