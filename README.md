# cURLinator

**AI-powered API testing and exploration platform**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-144%20passing-brightgreen.svg)](tests/)

---

## 🎯 Overview

**cURLinator** eliminates friction in API integration by providing a unified, conversational interface to discover, test, and execute API calls against any API documentation—regardless of format or quality.

### What is cURLinator?

cURLinator is an intelligent system that:

1. **Scrapes and understands** API documentation from any website
2. **Answers questions** about APIs using natural language
3. **Generates working cURL commands** from conversational queries
4. **Maintains context** across multiple questions for iterative exploration

### Key Features

- 🤖 **Two-Agent Architecture**: Specialized agents for documentation processing and conversational queries
- 🔍 **Smart OpenAPI Detection**: Automatically detects and parses OpenAPI/Swagger specifications
- 📚 **Intelligent Crawling**: Falls back to full-site crawling when specs aren't available
- 🧠 **RAG-Powered Queries**: Uses hybrid retrieval (semantic + BM25) for accurate answers
- 💬 **Conversation History**: Maintains context across queries with automatic summarization
- 🎯 **cURL Generation**: Produces ready-to-execute cURL commands from natural language
- 🗄️ **Local Vector Storage**: Uses Chroma for persistent, privacy-focused vector storage
- ⚡ **Contextual Enrichment**: Optional enrichment for more than 35% better retrieval accuracy

---

## 🏗️ Architecture

cURLinator uses a **simplified two-agent architecture**:

### 1. DocumentationAgent

**Purpose**: Scrapes and enriches API documentation  
**Input**: Base URL (e.g., `https://api.example.com/docs`)  
**Output**: `List[Document]` with structured metadata

**Features**:

- OpenAPI/Swagger spec detection (fast path)
- Full-site crawling with WholeSiteReader (fallback)
- Page classification (API reference, guides, tutorials, etc.)
- Optional contextual enrichment for better retrieval
- Headless browser support for JavaScript-heavy sites

### 2. ChatAgent

**Purpose**: Answers questions and generates cURL commands  
**Input**: Natural language query + conversation history  
**Output**: `{response: str, curl_command: str, sources: List[str]}`

**Features**:

- RAG-based query answering with hybrid retrieval
- Conversation history with automatic summarization
- Source citation for transparency
- Persistent Chroma vector database
- Configurable system prompts

---

## 📦 Installation

### Prerequisites

- **Python 3.11+**
- **LLM API Key** (at least one):
  - OpenAI API key (recommended)
  - Anthropic API key
  - Google Gemini API key

### Step 1: Clone the Repository

```bash
git clone https://github.com/kishorekumar1308/curlinator.git
cd curlinator
```

### Step 2: Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
pip install -e .
```

For development (includes pytest, black, ruff, mypy):

```bash
uv sync --all-extras
# or
pip install -e ".[dev]"
```

### Step 3: Configure Environment

Create a `.env` file in the project root:

```bash
# LLM Provider (choose one or more)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# LLM Configuration (optional)
LLM_PROVIDER=openai  # Options: openai, anthropic, gemini
LLM_MODEL=gpt-4o-mini  # Model name for your provider
LLM_TEMPERATURE=0.0  # Temperature for LLM responses

# Embedding Model (optional - defaults to local HuggingFace model)
EMBEDDING_PROVIDER=huggingface  # Options: openai, huggingface
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5  # Model name

# Vector Database (optional)
CHROMA_PERSIST_DIR=./chroma_db  # Where to store vector database
```

### Step 4: Verify Installation

```bash
uv run python examples/my_test.py
```

Expected output:

```
🚀 Starting crawl of: https://petstore3.swagger.io
✅ COMPLETE! Crawled 15 documents
📊 Document Types:
  - api_endpoint: 12
  - api_overview: 1
  - guide: 2
```

---

## 🚀 Quick Start

### Example 1: Scrape API Documentation

```python
import asyncio
from curlinator.agents import DocumentationAgent

async def main():
    # Initialize agent
    agent = DocumentationAgent(
        max_depth=3,           # Crawl depth
        max_pages=50,          # Max pages to crawl
        enable_enrichment=True, # Enable contextual enrichment
        headless=True          # Run browser in headless mode
    )

    # Scrape documentation
    documents = await agent.execute("https://petstore3.swagger.io")

    print(f"Crawled {len(documents)} documents")
    for doc in documents[:3]:
        print(f"- {doc.metadata.get('title', 'Untitled')}")

asyncio.run(main())
```

### Example 2: Query API Documentation

```python
import asyncio
from curlinator.agents import DocumentationAgent, ChatAgent

async def main():
    # Step 1: Scrape documentation
    doc_agent = DocumentationAgent(headless=True)
    documents = await doc_agent.execute("https://petstore3.swagger.io")

    # Step 2: Create chat agent
    chat_agent = ChatAgent(
        collection_name="petstore_api",
        persist_directory="./chroma_db"
    )

    # Step 3: Index documents
    await chat_agent.index_documents(documents)

    # Step 4: Ask questions
    result = await chat_agent.execute(
        "How do I create a new pet? Give me a cURL command."
    )

    print(f"Answer: {result['response']}")
    print(f"cURL: {result['curl_command']}")
    print(f"Sources: {result['sources']}")

asyncio.run(main())
```

### Example 3: Conversation with Follow-up Questions

```python
# First question
result1 = await chat_agent.execute("What endpoints are available for pets?")
print(result1['response'])

# Follow-up question (uses conversation history)
result2 = await chat_agent.execute("Show me how to update one")
print(result2['response'])
print(result2['curl_command'])

# Reset conversation if needed
chat_agent.reset_conversation()
```

---

## 🌐 REST API Documentation

cURLinator provides a FastAPI-based REST API for crawling documentation and querying indexed collections.

### Base URL

```
http://localhost:8000
```

### Authentication

Most endpoints require JWT authentication. Include the token in the `Authorization` header:

```bash
Authorization: Bearer <your_access_token>
```

### Rate Limiting

The API implements rate limiting to prevent abuse:

- **Auth endpoints** (`/register`, `/login`): 10 requests/minute per IP
- **Crawl endpoint** (`/crawl`): 5 requests/hour per IP (expensive operation)
- **Chat endpoint** (`/chat`): 60 requests/minute per IP
- **Collection endpoints**: No rate limiting

#### Rate Limit Headers

Rate-limited endpoints include the following headers in responses:

- **`X-RateLimit-Limit`**: Maximum number of requests allowed in the time window
- **`X-RateLimit-Remaining`**: Number of requests remaining in the current window
- **`X-RateLimit-Reset`**: Unix timestamp when the rate limit resets

Example response headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1699564800
```

#### Rate Limit Exceeded

When rate limit is exceeded, the API returns HTTP 429 with:

```json
{
  "error": "Rate limit exceeded",
  "message": "Too many requests. Please try again later.",
  "suggestion": "Wait a few minutes before making more requests."
}
```

The response also includes a `Retry-After` header indicating how many seconds to wait before retrying.

### Response Headers

All responses include:

- `X-Correlation-ID`: Unique request identifier for tracing
- `X-Process-Time`: Request processing time in seconds

Rate-limited endpoints also include:

- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when limit resets

---

### 🔐 Authentication Endpoints

#### Register a New User

```bash
POST /api/v1/auth/register
```

**Request Body:**

```json
{
  "email": "user@example.com",
  "password": "SecurePass123"
}
```

**Password Requirements:**

- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit

**Response (201 Created):**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "is_active": true
  }
}
```

**cURL Example:**

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123"
  }'
```

**Error Responses:**

- `400 Bad Request`: Email already registered
- `422 Unprocessable Entity`: Validation error (weak password, invalid email)
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Database error

---

#### Login

```bash
POST /api/v1/auth/login
```

**Request Body:**

```json
{
  "email": "user@example.com",
  "password": "SecurePass123"
}
```

**Response (200 OK):**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "is_active": true
  }
}
```

**cURL Example:**

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123"
  }'
```

**Error Responses:**

- `401 Unauthorized`: Invalid credentials
- `422 Unprocessable Entity`: Invalid email format
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Database error

---

### 📚 Crawl Endpoint

#### Crawl API Documentation

```bash
POST /api/v1/crawl
```

**Authentication Required:** Yes

**Request Body:**

```json
{
  "url": "https://petstore3.swagger.io",
  "max_pages": 50,
  "max_depth": 3,
  "embedding_provider": "LOCAL"
}
```

**Parameters:**

- `url` (required): Documentation URL to crawl
- `max_pages` (optional): Maximum pages to crawl (default: 50, max: 100)
- `max_depth` (optional): Maximum crawl depth (default: 3, max: 5)
- `embedding_provider` (optional): Embedding model provider
  - `LOCAL`: Local BAAI/bge-small-en-v1.5 (default, free)
  - `OPENAI`: OpenAI text-embedding-3-small (requires API key)
  - `GEMINI`: Google Gemini embedding (requires API key)
  - `AUTO`: Automatically selects best available provider

**URL Validation:**

- Must use HTTPS protocol (HTTP allowed for localhost)
- Cannot be localhost, private IPs, or cloud metadata endpoints
- Must be a valid, publicly accessible URL

**Response (200 OK):**

```json
{
  "crawl_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "pages_crawled": 15,
  "collection_name": "crawl_550e8400-e29b-41d4-a716-446655440000",
  "message": "Successfully crawled 15 pages",
  "embedding_provider": "LOCAL",
  "embedding_model": "BAAI/bge-small-en-v1.5"
}
```

**cURL Example:**

```bash
# Get access token first
TOKEN=$(curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"SecurePass123"}' \
  | jq -r '.access_token')

# Crawl documentation
curl -X POST http://localhost:8000/api/v1/crawl \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://petstore3.swagger.io",
    "max_pages": 50,
    "max_depth": 3,
    "embedding_provider": "LOCAL"
  }'
```

**Error Responses:**

- `400 Bad Request`: No documents found or invalid URL
- `403 Forbidden`: Not authenticated
- `408 Request Timeout`: Crawl operation timed out (>10 minutes)
- `422 Unprocessable Entity`: Validation error (invalid URL, parameters out of range)
- `429 Too Many Requests`: Rate limit exceeded (5/hour)
- `500 Internal Server Error`: Crawl failed or embedding model error

---

### 💬 Chat Endpoint

#### Query Indexed Documentation

```bash
POST /api/v1/chat
```

**Authentication Required:** Yes

**Request Body:**

```json
{
  "collection_name": "crawl_550e8400-e29b-41d4-a716-446655440000",
  "message": "How do I create a new pet? Give me a cURL command.",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Parameters:**

- `collection_name` (required): Collection name from crawl response
- `message` (required): Natural language query
- `session_id` (optional): Chat session ID for conversation history. If not provided, a new session will be created. Use this to maintain conversation context across multiple queries.
- `conversation_history` (optional, deprecated): Previous conversation messages. Use `session_id` instead for server-side history management.

**Response (200 OK):**

```json
{
  "response": "To create a new pet, you can use the POST /pet endpoint...",
  "curl_command": "curl -X POST https://petstore3.swagger.io/api/v3/pet -H 'Content-Type: application/json' -d '{\"name\":\"doggie\",\"status\":\"available\"}'",
  "sources": [
    "https://petstore3.swagger.io/api/v3/pet",
    "https://petstore3.swagger.io/docs/pet"
  ],
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Conversation History:**

The chat endpoint now supports server-side conversation history management through sessions:

1. **First message**: Send a message without `session_id`. The response will include a new `session_id`.
2. **Follow-up messages**: Include the `session_id` from the previous response to maintain conversation context.
3. **Session management**: Use the session management endpoints below to list, view, delete, or reset sessions.

**Example - Multi-turn conversation:**

```bash
# First message - creates new session
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_collection",
    "message": "What endpoints are available?"
  }'

# Response includes session_id
{
  "response": "The API has the following endpoints...",
  "session_id": "abc123...",
  ...
}

# Follow-up message - uses existing session
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "my_collection",
    "message": "Show me a cURL example for the first one",
    "session_id": "abc123..."
  }'
```

**cURL Example:**

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "crawl_550e8400-e29b-41d4-a716-446655440000",
    "message": "How do I create a new pet? Give me a cURL command."
  }'
```

**Error Responses:**

- `403 Forbidden`: Not authenticated or collection not owned by user
- `404 Not Found`: Collection or session not found
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded (60/minute)
- `500 Internal Server Error`: Query failed

---

### 💬 Session Management Endpoints

#### List Chat Sessions

```bash
GET /api/v1/sessions
```

**Authentication Required:** Yes

**Description:** List all chat sessions for the authenticated user, ordered by most recently updated.

**Response (200 OK):**

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "collection_name": "my_api_docs",
    "collection_id": "abc123...",
    "message_count": 8,
    "created_at": "2025-11-15T10:30:00Z",
    "updated_at": "2025-11-15T11:45:00Z"
  }
]
```

**cURL Example:**

```bash
curl -X GET http://localhost:8000/api/v1/sessions \
  -H "Authorization: Bearer $TOKEN"
```

---

#### Get Session Detail

```bash
GET /api/v1/sessions/{session_id}
```

**Authentication Required:** Yes

**Description:** Get detailed information about a specific chat session including all messages.

**Response (200 OK):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "collection_name": "my_api_docs",
  "collection_id": "abc123...",
  "messages": [
    {
      "id": "msg1...",
      "role": "user",
      "content": "What endpoints are available?",
      "curl_command": null,
      "created_at": "2025-11-15T10:30:00Z"
    },
    {
      "id": "msg2...",
      "role": "assistant",
      "content": "The API has the following endpoints...",
      "curl_command": "curl -X GET https://api.example.com/endpoints",
      "created_at": "2025-11-15T10:30:05Z"
    }
  ],
  "created_at": "2025-11-15T10:30:00Z",
  "updated_at": "2025-11-15T10:30:05Z"
}
```

**cURL Example:**

```bash
curl -X GET http://localhost:8000/api/v1/sessions/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer $TOKEN"
```

**Error Responses:**

- `404 Not Found`: Session not found or you don't have access to it

---

#### Delete Session

```bash
DELETE /api/v1/sessions/{session_id}
```

**Authentication Required:** Yes

**Description:** Delete a chat session and all its messages permanently.

**Response (200 OK):**

```json
{
  "message": "Session deleted successfully",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**cURL Example:**

```bash
curl -X DELETE http://localhost:8000/api/v1/sessions/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer $TOKEN"
```

**Error Responses:**

- `404 Not Found`: Session not found or you don't have access to it

---

#### Reset Session

```bash
POST /api/v1/sessions/{session_id}/reset
```

**Authentication Required:** Yes

**Description:** Clear all messages from a chat session while keeping the session itself. Useful for starting a fresh conversation on the same collection.

**Response (200 OK):**

```json
{
  "message": "Session reset successfully",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "messages_deleted": 8
}
```

**cURL Example:**

```bash
curl -X POST http://localhost:8000/api/v1/sessions/550e8400-e29b-41d4-a716-446655440000/reset \
  -H "Authorization: Bearer $TOKEN"
```

**Error Responses:**

- `404 Not Found`: Session not found or you don't have access to it

---

### 📁 Collection Management Endpoints

#### List Collections

```bash
GET /api/v1/collections
```

**Authentication Required:** Yes

**Response (200 OK):**

```json
{
  "collections": [
    {
      "name": "crawl_550e8400-e29b-41d4-a716-446655440000",
      "base_url": "https://petstore3.swagger.io",
      "pages_crawled": 15,
      "created_at": "2025-11-15T10:30:00Z"
    }
  ],
  "total": 1
}
```

**cURL Example:**

```bash
curl -X GET http://localhost:8000/api/v1/collections \
  -H "Authorization: Bearer $TOKEN"
```

---

#### Get Collection Details

```bash
GET /api/v1/collections/{collection_name}
```

**Authentication Required:** Yes

**Response (200 OK):**

```json
{
  "name": "crawl_550e8400-e29b-41d4-a716-446655440000",
  "base_url": "https://petstore3.swagger.io",
  "pages_crawled": 15,
  "created_at": "2025-11-15T10:30:00Z",
  "embedding_provider": "LOCAL",
  "embedding_model": "BAAI/bge-small-en-v1.5"
}
```

**cURL Example:**

```bash
curl -X GET http://localhost:8000/api/v1/collections/crawl_550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer $TOKEN"
```

**Error Responses:**

- `403 Forbidden`: Not authenticated or collection not owned by user
- `404 Not Found`: Collection not found

---

#### Delete Collection

```bash
DELETE /api/v1/collections/{collection_name}
```

**Authentication Required:** Yes

**Response (200 OK):**

```json
{
  "message": "Collection deleted successfully",
  "collection_name": "crawl_550e8400-e29b-41d4-a716-446655440000"
}
```

**cURL Example:**

```bash
curl -X DELETE http://localhost:8000/api/v1/collections/crawl_550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer $TOKEN"
```

**Error Responses:**

- `403 Forbidden`: Not authenticated or collection not owned by user
- `404 Not Found`: Collection not found
- `500 Internal Server Error`: Failed to delete collection

---

### 🤝 Collection Sharing Endpoints

cURLinator allows you to share collections with other users and control their access permissions. Collections can be private (owner-only), shared with specific users, or made public for everyone.

#### Permission Levels

- **`view`**: User can view collection metadata only (name, URL, page count, etc.)
- **`chat`**: User can view collection metadata AND query the collection via the chat endpoint

#### Visibility Options

- **`private`**: Only the owner can access the collection (default)
- **`public`**: Anyone can view and chat with the collection (no sharing needed)

---

#### Share Collection with User

```bash
POST /api/v1/collections/{collection_name}/share
```

**Authentication Required:** Yes (must be collection owner)

**Description:** Share a collection with another user by email address.

**Request Body:**

```json
{
  "user_email": "colleague@example.com",
  "permission": "chat"
}
```

**Parameters:**

- `user_email` (required): Email address of the user to share with
- `permission` (required): Permission level - `"view"` or `"chat"`

**Response (201 Created):**

```json
{
  "id": "share_550e8400...",
  "collection_id": "abc123...",
  "collection_name": "my_api_docs",
  "user_id": "user456...",
  "user_email": "colleague@example.com",
  "permission": "chat",
  "created_at": "2025-11-15T10:30:00Z",
  "updated_at": "2025-11-15T10:30:00Z"
}
```

**cURL Example:**

```bash
curl -X POST http://localhost:8000/api/v1/collections/my_api_docs/share \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_email": "colleague@example.com",
    "permission": "chat"
  }'
```

**Error Responses:**

- `403 Forbidden`: Not authenticated or not the collection owner
- `404 Not Found`: Collection or target user not found
- `400 Bad Request`: Invalid permission level or trying to share with yourself

---

#### List Collection Shares

```bash
GET /api/v1/collections/{collection_name}/shares
```

**Authentication Required:** Yes (must be collection owner)

**Description:** List all users who have access to a collection.

**Response (200 OK):**

```json
[
  {
    "id": "share_550e8400...",
    "collection_id": "abc123...",
    "collection_name": "my_api_docs",
    "user_id": "user456...",
    "user_email": "colleague@example.com",
    "permission": "chat",
    "created_at": "2025-11-15T10:30:00Z",
    "updated_at": "2025-11-15T10:30:00Z"
  }
]
```

**cURL Example:**

```bash
curl -X GET http://localhost:8000/api/v1/collections/my_api_docs/shares \
  -H "Authorization: Bearer $TOKEN"
```

**Error Responses:**

- `403 Forbidden`: Not authenticated or not the collection owner
- `404 Not Found`: Collection not found

---

#### Update Share Permission

```bash
PATCH /api/v1/collections/{collection_name}/shares/{user_email}
```

**Authentication Required:** Yes (must be collection owner)

**Description:** Update the permission level for an existing share.

**Request Body:**

```json
{
  "permission": "view"
}
```

**Parameters:**

- `permission` (required): New permission level - `"view"` or `"chat"`

**Response (200 OK):**

```json
{
  "id": "share_550e8400...",
  "collection_id": "abc123...",
  "collection_name": "my_api_docs",
  "user_id": "user456...",
  "user_email": "colleague@example.com",
  "permission": "view",
  "created_at": "2025-11-15T10:30:00Z",
  "updated_at": "2025-11-15T10:35:00Z"
}
```

**cURL Example:**

```bash
curl -X PATCH http://localhost:8000/api/v1/collections/my_api_docs/shares/colleague@example.com \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "permission": "view"
  }'
```

**Error Responses:**

- `403 Forbidden`: Not authenticated or not the collection owner
- `404 Not Found`: Collection or share not found

---

#### Revoke Share

```bash
DELETE /api/v1/collections/{collection_name}/shares/{user_email}
```

**Authentication Required:** Yes (must be collection owner)

**Description:** Revoke a user's access to a shared collection.

**Response (204 No Content):**

No response body.

**cURL Example:**

```bash
curl -X DELETE http://localhost:8000/api/v1/collections/my_api_docs/shares/colleague@example.com \
  -H "Authorization: Bearer $TOKEN"
```

**Error Responses:**

- `403 Forbidden`: Not authenticated or not the collection owner
- `404 Not Found`: Collection or share not found

---

#### Update Collection Visibility

```bash
PATCH /api/v1/collections/{collection_name}/visibility
```

**Authentication Required:** Yes (must be collection owner)

**Description:** Update the visibility of a collection (private or public).

**Request Body:**

```json
{
  "visibility": "public"
}
```

**Parameters:**

- `visibility` (required): Visibility level - `"private"` or `"public"`

**Response (200 OK):**

```json
{
  "name": "my_api_docs",
  "base_url": "https://api.example.com/docs",
  "pages_crawled": 25,
  "created_at": "2025-11-15T10:00:00Z",
  "visibility": "public"
}
```

**cURL Example:**

```bash
curl -X PATCH http://localhost:8000/api/v1/collections/my_api_docs/visibility \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "visibility": "public"
  }'
```

**Error Responses:**

- `403 Forbidden`: Not authenticated or not the collection owner
- `404 Not Found`: Collection not found

---

### 🏥 Health Check

#### Check API Health

```bash
GET /health
```

**Authentication Required:** No

**Response (200 OK):**

```json
{
  "status": "healthy",
  "timestamp": "2025-11-15T10:30:00Z",
  "version": "1.0.0",
  "checks": {
    "database": {
      "status": "healthy",
      "message": "Database connection successful"
    },
    "chroma": {
      "status": "healthy",
      "message": "Chroma connection successful",
      "collections_count": 5
    },
    "system": {
      "status": "healthy",
      "cpu_percent": 15.2,
      "memory_percent": 45.8,
      "memory_available_gb": 8.5,
      "disk_percent": 62.3,
      "disk_free_gb": 120.5,
      "warnings": null
    },
    "llm": {
      "openai": {
        "status": "healthy",
        "message": "API connection successful"
      },
      "gemini": {
        "status": "not_configured",
        "message": "API key not set"
      },
      "anthropic": {
        "status": "healthy",
        "message": "API key configured (connectivity not tested to save costs)"
      }
    }
  }
}
```

**Health Status Values:**

- `healthy`: All systems operational
- `degraded`: System operational but with warnings (high CPU/memory/disk usage or LLM API connectivity issues)
- `unhealthy`: Critical system failure (database or Chroma unavailable)

**LLM API Status Values:**

- `healthy`: API key configured and connectivity verified
- `degraded`: API key configured but connectivity test failed
- `not_configured`: API key not set

**System Warnings:**

The health check monitors system resources and returns warnings when:

- CPU usage > 90%
- Memory usage > 90%
- Disk usage > 90%

**LLM Connectivity Checks:**

The health check performs lightweight connectivity tests for configured LLM APIs:

- **OpenAI**: Lists available models to verify API access
- **Gemini**: Lists available models to verify API access
- **Anthropic**: Validates API key configuration (no actual API call to save costs)

These checks are **non-blocking** - LLM API failures will mark the service as `degraded` but not `unhealthy`, allowing the API to continue serving requests even if LLM providers are temporarily unavailable.

**cURL Example:**

```bash
curl -X GET http://localhost:8000/health
```

---

### 📊 Error Response Format

All error responses follow a consistent structure with standardized error codes:

```json
{
  "error_code": "AUTH_INVALID_CREDENTIALS",
  "error": "Auth Invalid Credentials",
  "message": "Invalid email or password.",
  "suggestion": "Please check your credentials and try again."
}
```

**Error Response Fields:**

- `error_code`: Machine-readable error code for programmatic handling
- `error`: Human-readable error type (derived from error_code)
- `message`: Detailed error message
- `suggestion`: Actionable suggestion to fix the error (optional)

**Error Code Categories:**

- `AUTH_*`: Authentication and authorization errors
- `VALIDATION_*`: Input validation errors
- `RESOURCE_*`: Resource-related errors (not found, conflict, etc.)
- `CRAWL_*`: Crawling operation errors
- `CHAT_*`: Chat/query operation errors
- `DATABASE_*`: Database operation errors
- `RATE_LIMIT_*`: Rate limiting errors
- `SYSTEM_*`: System/server errors

**Common Error Codes:**

- `AUTH_INVALID_CREDENTIALS`: Invalid email or password
- `AUTH_NOT_AUTHENTICATED`: Missing or invalid authentication token
- `RESOURCE_ALREADY_EXISTS`: Resource already exists (e.g., duplicate email)
- `RESOURCE_NOT_FOUND`: Resource not found
- `VALIDATION_INVALID_URL`: Invalid URL format
- `VALIDATION_WEAK_PASSWORD`: Password doesn't meet strength requirements
- `CRAWL_TIMEOUT`: Crawl operation timed out
- `CRAWL_NO_DOCUMENTS`: No documents found during crawl
- `DATABASE_INTEGRITY_ERROR`: Database constraint violation
- `DATABASE_QUERY_FAILED`: Database query failed
- `RATE_LIMIT_EXCEEDED`: Rate limit exceeded

**Common HTTP Status Codes:**

- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Authenticated but not authorized
- `404 Not Found`: Resource not found
- `408 Request Timeout`: Operation timed out
- `409 Conflict`: Resource conflict (e.g., duplicate email)
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

---

## 📖 Python API Reference

### DocumentationAgent

```python
class DocumentationAgent:
    def __init__(
        self,
        max_depth: int = 3,              # Maximum crawl depth
        max_pages: int = 50,             # Maximum pages to crawl
        enable_enrichment: bool = True,  # Enable contextual enrichment
        use_llm_classification: bool = False,  # Use LLM for page classification
        headless: bool = True,           # Run browser in headless mode
        verbose: bool = False            # Enable verbose logging
    )

    async def execute(self, base_url: str) -> List[Document]:
        """
        Scrape and process API documentation.

        Returns:
            List[Document]: Documents with metadata (title, type, URL, etc.)
        """
```

**Document Metadata Fields**:

- `title`: Page title
- `type` or `page_type`: Page classification (api_endpoint, guide, tutorial, etc.)
- `URL` or `url` or `source`: Page URL
- `description`: Page description
- `headings`: List of headings on the page
- `enriched`: Whether contextual enrichment was applied

### ChatAgent

```python
class ChatAgent:
    def __init__(
        self,
        collection_name: str,                    # Chroma collection name
        persist_directory: str = "./chroma_db",  # Vector DB directory
        system_prompt: Optional[str] = None,     # Custom system prompt
        verbose: bool = False                    # Enable verbose logging
    )

    async def index_documents(self, documents: List[Document]) -> None:
        """Index documents into vector database."""

    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Answer a question about the API.

        Returns:
            {
                "response": str,        # Natural language answer
                "curl_command": str,    # Generated cURL command (if applicable)
                "sources": List[str]    # Source URLs used
            }
        """

    def reset_conversation(self) -> None:
        """Clear conversation history."""
```

---

## 🧪 Testing

cURLinator has comprehensive test coverage with **144 passing tests**:

- **121 unit tests** (95% code coverage)
- **23 integration tests** (end-to-end validation)

### Run All Tests

```bash
uv run pytest
```

### Run Unit Tests Only

```bash
uv run pytest tests/unit tests/agents tests/utils -v
```

### Run Integration Tests

```bash
# Requires LLM API key
uv run pytest tests/integration -v
```

### Run with Coverage

```bash
uv run pytest --cov=src/curlinator --cov-report=html
open htmlcov/index.html
```

### Test Markers

```bash
# Skip slow tests
uv run pytest -m "not slow"

# Skip integration tests
uv run pytest -m "not integration"

# Run only integration tests
uv run pytest -m integration
```

---

## 📊 Monitoring and Metrics

cURLinator includes built-in monitoring and observability features:

- **Prometheus Metrics** - Performance and health metrics
- **Sentry Error Tracking** - Real-time error monitoring and alerting
- **Structured Logging** - Comprehensive application logs with correlation IDs

### 🔍 Sentry Error Tracking

Sentry integration provides real-time error monitoring, performance tracking, and alerting.

#### Setup

1. **Create a Sentry account** at [sentry.io](https://sentry.io)
2. **Create a new project** for cURLinator (select "FastAPI" as the platform)
3. **Copy your DSN** from the project settings
4. **Add to your `.env` file**:

```bash
# Sentry Configuration
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
SENTRY_ENVIRONMENT=production  # Optional: defaults to ENVIRONMENT value
SENTRY_TRACES_SAMPLE_RATE=0.1  # Optional: 10% of requests (default)
```

#### Features

- ✅ **Automatic error capture** - All unhandled exceptions are reported
- ✅ **Performance monitoring** - Track slow endpoints and database queries
- ✅ **Request context** - Full request details attached to errors
- ✅ **User context** - User ID and email attached to errors (when authenticated)
- ✅ **Database query tracking** - SQLAlchemy integration for query monitoring
- ✅ **Release tracking** - Track errors by application version
- ✅ **Environment separation** - Separate errors by development/staging/production

#### What Gets Tracked

- **5xx Server Errors** - All internal server errors
- **Unhandled Exceptions** - Any exception not caught by application code
- **Database Errors** - Connection failures, query errors, constraint violations
- **LLM API Errors** - OpenAI/Gemini/Anthropic API failures
- **Performance Issues** - Slow endpoints, N+1 queries, memory issues

#### Disabling Sentry

To disable Sentry (e.g., for local development), simply leave `SENTRY_DSN` empty or remove it from your `.env` file:

```bash
# Sentry disabled
SENTRY_DSN=
```

### 📈 Prometheus Metrics

cURLinator includes built-in Prometheus metrics for monitoring application health and performance.

#### Metrics Endpoint

The `/metrics` endpoint exposes Prometheus-formatted metrics (no authentication required):

```bash
curl http://localhost:8000/metrics
```

#### Available Metrics

**HTTP Metrics**

- **`curlinator_http_requests_total`** - Total HTTP requests by method, endpoint, and status code
- **`curlinator_http_request_duration_seconds`** - HTTP request latency histogram
- **`curlinator_http_requests_in_progress`** - Current number of in-progress requests

**Authentication Metrics**

- **`curlinator_auth_attempts_total`** - Authentication attempts by endpoint and status
- **`curlinator_auth_tokens_created_total`** - Total JWT tokens created

**Database Metrics**

- **`curlinator_db_queries_total`** - Database queries by operation, table, and status
- **`curlinator_db_query_duration_seconds`** - Database query latency histogram
- **`curlinator_db_connections_active`** - Active database connections
- **`curlinator_db_errors_total`** - Database errors by error type

**Vector Store Metrics**

- **`curlinator_vectorstore_operations_total`** - Vector store operations by type and status
- **`curlinator_vectorstore_operation_duration_seconds`** - Vector store operation latency
- **`curlinator_vectorstore_documents_indexed_total`** - Total documents indexed
- **`curlinator_vectorstore_queries_total`** - Vector store queries by collection

**Crawling Metrics**

- **`curlinator_crawl_operations_total`** - Crawl operations by status
- **`curlinator_crawl_duration_seconds`** - Crawl duration histogram
- **`curlinator_crawl_pages_total`** - Pages crawled by status
- **`curlinator_crawl_pages_per_operation`** - Pages per crawl operation histogram

**Chat/Query Metrics**

- **`curlinator_chat_queries_total`** - Chat queries by collection and status
- **`curlinator_chat_query_duration_seconds`** - Chat query latency histogram
- **`curlinator_chat_messages_total`** - Chat messages by role (user/assistant)

**Collection Sharing Metrics**

- **`curlinator_collection_shares_total`** - Collection share operations
- **`curlinator_collection_visibility_changes_total`** - Collection visibility changes

**Application Info**

- **`curlinator_app_info`** - Application metadata (version, environment)

#### Prometheus Configuration

Example `prometheus.yml` configuration for scraping cURLinator metrics:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "curlinator"
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: "/metrics"
```

### Example Queries

**Request rate by endpoint:**

```promql
rate(curlinator_http_requests_total[5m])
```

**95th percentile request latency:**

```promql
histogram_quantile(0.95, rate(curlinator_http_request_duration_seconds_bucket[5m]))
```

**Failed crawl operations:**

```promql
curlinator_crawl_operations_total{status="failure"}
```

**Active database connections:**

```promql
curlinator_db_connections_active
```

**Chat query success rate:**

```promql
rate(curlinator_chat_queries_total{status="success"}[5m]) / rate(curlinator_chat_queries_total[5m])
```

### Grafana Dashboard

You can create a Grafana dashboard to visualize these metrics. Key panels to include:

- **Request Rate**: `rate(curlinator_http_requests_total[5m])`
- **Error Rate**: `rate(curlinator_http_requests_total{status_code=~"5.."}[5m])`
- **Request Latency**: `histogram_quantile(0.95, rate(curlinator_http_request_duration_seconds_bucket[5m]))`
- **Active Connections**: `curlinator_db_connections_active`
- **Crawl Success Rate**: `rate(curlinator_crawl_operations_total{status="success"}[5m])`
- **Vector Store Operations**: `rate(curlinator_vectorstore_operations_total[5m])`

---

## 🔄 Database Connection Resilience

### Automatic Retry Logic

cURLinator includes automatic database connection retry logic with exponential backoff. This is especially important for cloud deployments (like Railway) where the database may start after the API service.

**Features:**

- **Exponential Backoff**: Retries with increasing delays (1s, 2s, 4s, 8s, 16s)
- **Configurable Retries**: Set `DATABASE_MAX_RETRIES` environment variable (default: 5)
- **Detailed Logging**: Logs each retry attempt and final success/failure
- **Graceful Failure**: Raises clear error after all retries exhausted

**Configuration:**

```bash
# .env file
DATABASE_MAX_RETRIES=5  # Maximum retry attempts (default: 5)
```

**Example Logs:**

```
WARNING:curlinator.api.database:⚠️  Database connection failed (attempt 1/5). Retrying in 1s...
WARNING:curlinator.api.database:⚠️  Database connection failed (attempt 2/5). Retrying in 2s...
INFO:curlinator.api.database:✅ Database connection established after 2 retries
```

**Retry Schedule:**

| Attempt | Wait Time | Total Time Elapsed |
| ------- | --------- | ------------------ |
| 1       | 0s        | 0s                 |
| 2       | 1s        | 1s                 |
| 3       | 2s        | 3s                 |
| 4       | 4s        | 7s                 |
| 5       | 8s        | 15s                |
| 6       | 16s       | 31s                |

**Why This Matters:**

In cloud environments like Railway, services may start in parallel. The database container might take a few seconds to become ready while the API service is already starting. Without retry logic, the API would crash immediately. With retry logic, the API waits for the database to become available, ensuring smooth deployments.

---

## 💾 Database Backup & Recovery

### Overview

cURLinator uses PostgreSQL for persistent data storage. Regular backups are essential to prevent data loss and ensure business continuity. This section covers backup strategies, commands, and recovery procedures.

### What Gets Backed Up

The PostgreSQL database contains:

- **User accounts** and authentication data
- **Documentation collections** metadata (URLs, crawl settings, embedding models)
- **Chat sessions** and conversation history
- **Collection sharing** permissions and access control

**Note**: The Chroma vector database is stored separately in the `chroma_db/` directory and should also be backed up.

---

### Backup Strategies

#### 1. Manual Backup (Development)

For local development and testing:

```bash
# Backup database to a file
pg_dump -h localhost -U curlinator -d curlinator_db -F c -f backup_$(date +%Y%m%d_%H%M%S).dump

# With Docker Compose
docker-compose exec postgres pg_dump -U curlinator -d curlinator_db -F c > backup_$(date +%Y%m%d_%H%M%S).dump
```

**Options explained**:

- `-F c`: Custom format (compressed, supports parallel restore)
- `-f`: Output file name
- `$(date +%Y%m%d_%H%M%S)`: Timestamp in filename

#### 2. Automated Backup (Production)

**Recommended Schedule**:

- **Full backups**: Daily at 2 AM (low traffic period)
- **Incremental backups**: Every 6 hours
- **Retention policy**: Keep 7 daily, 4 weekly, 12 monthly backups

**Example cron job** (add to crontab with `crontab -e`):

```bash
# Daily full backup at 2 AM
0 2 * * * pg_dump -h localhost -U curlinator -d curlinator_db -F c -f /backups/curlinator_$(date +\%Y\%m\%d).dump && find /backups -name "curlinator_*.dump" -mtime +7 -delete

# Backup Chroma vector database
0 2 * * * tar -czf /backups/chroma_$(date +\%Y\%m\%d).tar.gz /app/chroma_db/
```

#### 3. Cloud Backup (Production)

For production deployments, use managed backup solutions:

**AWS RDS**:

```bash
# Automated backups are enabled by default
# Manual snapshot
aws rds create-db-snapshot \
  --db-instance-identifier curlinator-prod \
  --db-snapshot-identifier curlinator-manual-$(date +%Y%m%d)
```

**Google Cloud SQL**:

```bash
# Create on-demand backup
gcloud sql backups create \
  --instance=curlinator-prod \
  --description="Manual backup $(date +%Y-%m-%d)"
```

**Azure Database for PostgreSQL**:

```bash
# Automated backups are enabled by default (7-35 day retention)
# Point-in-time restore available
```

---

### Backup Verification

Always verify backups are valid:

```bash
# List contents of backup file
pg_restore -l backup_20241115_140000.dump

# Test restore to a temporary database
createdb curlinator_test
pg_restore -d curlinator_test backup_20241115_140000.dump
psql -d curlinator_test -c "SELECT COUNT(*) FROM users;"
dropdb curlinator_test
```

---

### Recovery Procedures

#### 1. Full Database Restore

**⚠️ WARNING**: This will **overwrite all existing data**. Ensure you have a recent backup before proceeding.

```bash
# Step 1: Stop the API server
docker-compose stop api

# Step 2: Drop and recreate the database
docker-compose exec postgres psql -U curlinator -c "DROP DATABASE curlinator_db;"
docker-compose exec postgres psql -U curlinator -c "CREATE DATABASE curlinator_db;"

# Step 3: Restore from backup
docker-compose exec -T postgres pg_restore -U curlinator -d curlinator_db < backup_20241115_140000.dump

# Step 4: Verify data
docker-compose exec postgres psql -U curlinator -d curlinator_db -c "SELECT COUNT(*) FROM users;"

# Step 5: Restart the API server
docker-compose start api
```

#### 2. Selective Table Restore

To restore specific tables without affecting others:

```bash
# Restore only the users table
pg_restore -d curlinator_db -t users backup_20241115_140000.dump

# Restore multiple tables
pg_restore -d curlinator_db -t users -t documentation_collections backup_20241115_140000.dump
```

#### 3. Point-in-Time Recovery (PITR)

For production systems with continuous archiving:

```bash
# Restore to specific timestamp
pg_restore -d curlinator_db --target-time="2024-11-15 14:30:00" backup_20241115_140000.dump
```

#### 4. Restore Chroma Vector Database

```bash
# Stop the API server
docker-compose stop api

# Extract Chroma backup
tar -xzf chroma_20241115.tar.gz -C /app/

# Restart the API server
docker-compose start api
```

---

### Disaster Recovery Checklist

In case of data loss or corruption:

- [ ] **Stop the API server** to prevent further data changes
- [ ] **Identify the most recent valid backup** (check backup verification logs)
- [ ] **Estimate data loss window** (time between backup and incident)
- [ ] **Notify stakeholders** about the incident and estimated recovery time
- [ ] **Restore database** from backup (see procedures above)
- [ ] **Restore Chroma vector database** from backup
- [ ] **Run database migrations** if needed: `alembic upgrade head`
- [ ] **Verify data integrity** (check user count, collection count, session count)
- [ ] **Test critical functionality** (login, chat, crawl)
- [ ] **Restart API server** and monitor logs
- [ ] **Document the incident** and update backup procedures if needed

---

### Backup Best Practices

1. **Test restores regularly** - Backups are useless if they can't be restored
2. **Store backups off-site** - Use cloud storage (S3, GCS, Azure Blob)
3. **Encrypt backups** - Use `pg_dump` with encryption or encrypt at rest
4. **Monitor backup jobs** - Set up alerts for failed backups
5. **Document procedures** - Keep this guide updated with your specific setup
6. **Automate everything** - Manual backups are error-prone
7. **Version control migrations** - Keep all Alembic migration files in Git

---

### Backup Storage Recommendations

**Local Development**:

- Store in `/backups` directory (excluded from Git)
- Keep last 7 days of backups

**Production**:

- **Primary**: Cloud storage (S3, GCS, Azure Blob) with versioning enabled
- **Secondary**: Different cloud provider or region (disaster recovery)
- **Retention**: 7 daily + 4 weekly + 12 monthly backups
- **Encryption**: Enable encryption at rest and in transit

**Example S3 backup script**:

```bash
#!/bin/bash
# backup-to-s3.sh

BACKUP_FILE="curlinator_$(date +%Y%m%d_%H%M%S).dump"
S3_BUCKET="s3://your-backup-bucket/curlinator/"

# Create backup
pg_dump -h localhost -U curlinator -d curlinator_db -F c -f "/tmp/$BACKUP_FILE"

# Upload to S3
aws s3 cp "/tmp/$BACKUP_FILE" "$S3_BUCKET" --storage-class STANDARD_IA

# Clean up local file
rm "/tmp/$BACKUP_FILE"

# Delete backups older than 30 days from S3
aws s3 ls "$S3_BUCKET" | while read -r line; do
    createDate=$(echo $line | awk '{print $1" "$2}')
    createDate=$(date -d "$createDate" +%s)
    olderThan=$(date -d "30 days ago" +%s)
    if [[ $createDate -lt $olderThan ]]; then
        fileName=$(echo $line | awk '{print $4}')
        if [[ $fileName != "" ]]; then
            aws s3 rm "$S3_BUCKET$fileName"
        fi
    fi
done
```

---

## 🛠️ Development

### Project Structure

```
curlinator/
├── src/curlinator/
│   ├── agents/              # DocumentationAgent, ChatAgent
│   ├── utils/               # OpenAPI detector, page classifier, enrichment
│   ├── core/                # OpenAPI validator
│   └── config.py            # Settings and configuration
├── tests/
│   ├── unit/                # Unit tests for utilities
│   ├── agents/              # Unit tests for agents
│   └── integration/         # Integration tests
├── examples/                # Example scripts
└── pyproject.toml           # Project configuration
```

### Code Quality Tools

```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type checking
uv run mypy src/
```

### Running Examples

```bash
# Minimal example (Swagger Petstore)
uv run python examples/my_test.py

# Full DocumentationAgent example
uv run python examples/test_documentation_agent.py

# End-to-end workflow example
uv run python examples/test_end_to_end_workflow.py
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with tests
4. **Run tests**: `uv run pytest`
5. **Format code**: `uv run black src/ tests/`
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Development Setup

```bash
# Clone the repository (or your fork)
git clone https://github.com/kishorekumar1308/curlinator.git
cd curlinator

# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Format and lint
uv run black src/ tests/
uv run ruff check src/ tests/
```

---

## Acknowledgments

Built with:

- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [Chroma](https://www.trychroma.com/) - Vector database
- [OpenAI](https://openai.com/) / [Anthropic](https://www.anthropic.com/) / [Google Gemini](https://ai.google.dev/) - LLM providers
- [Selenium](https://www.selenium.dev/) - Web automation
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) - HTML parsing
