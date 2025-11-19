# cURLinator API Setup Guide

This guide will help you set up and run the cURLinator FastAPI backend locally.

## Prerequisites

- Python 3.11+
- PostgreSQL 14+ (for authentication and database features)
- Chrome/Chromium (for Selenium-based documentation crawling)

## Installation

### 1. Install Dependencies

```bash
# Install server dependencies (includes FastAPI, SQLAlchemy, etc.)
uv sync --extra server

# Or with pip
pip install -e ".[server]"
```

### 2. Set Up PostgreSQL Database

```bash
# Create database
createdb curlinator_dev

# Or using psql
psql -U postgres -c "CREATE DATABASE curlinator_dev;"
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Database
DATABASE_URL=postgresql://localhost:5432/curlinator_dev

# CORS (comma-separated list of allowed origins)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# JWT Authentication
JWT_SECRET=your-super-secret-key-change-in-production

# LLM API Keys (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### 4. Run Database Migrations

```bash
# Create initial migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head
```

## Running the API

### Development Server

```bash
# Run with auto-reload
uvicorn curlinator.api.main:app --reload --port 8000

# Or using the main.py script
python -m curlinator.api.main
```

The API will be available at:

- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Authentication

**Register:**

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123"
  }'
```

**Login:**

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123"
  }'
```

**Get Current User:**

```bash
TOKEN="your-jwt-token-here"

curl http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

### Crawl Documentation

```bash
TOKEN="your-jwt-token-here"

curl -X POST http://localhost:8000/api/v1/crawl \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "url": "https://stripe.com/docs/api",
    "max_pages": 10,
    "max_depth": 3
  }'
```

Response:

```json
{
  "crawl_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "pages_crawled": 10,
  "collection_name": "crawl_123e4567-e89b-12d3-a456-426614174000",
  "message": "Successfully crawled 10 pages"
}
```

### Query Documentation

```bash
TOKEN="your-jwt-token-here"
COLLECTION="crawl_123e4567-e89b-12d3-a456-426614174000"

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "{
    \"collection_name\": \"$COLLECTION\",
    \"message\": \"How do I create a customer?\",
    \"conversation_history\": []
  }"
```

Response:

```json
{
  "response": "To create a customer, use the POST /v1/customers endpoint...",
  "curl_command": "curl https://api.stripe.com/v1/customers -u sk_test_xxx: -d email=customer@example.com",
  "sources": [
    {
      "rank": 1,
      "score": 0.85,
      "text": "Create a customer object...",
      "url": "https://stripe.com/docs/api/customers/create"
    }
  ]
}
```

## Testing

### Manual Testing with Swagger UI

1. Open http://localhost:8000/docs
2. Click "Authorize" button
3. Register a new user via `/api/v1/auth/register`
4. Copy the `access_token` from the response
5. Click "Authorize" again and paste the token
6. Test the `/api/v1/crawl` and `/api/v1/chat` endpoints

### Testing with curl

See the examples above for curl commands.

## Database Management

### View Tables

```bash
psql curlinator_dev -c "\dt"
```

### View Users

```bash
psql curlinator_dev -c "SELECT id, email, created_at FROM users;"
```

### View Collections

```bash
psql curlinator_dev -c "SELECT id, name, domain, pages_crawled, owner_id FROM documentation_collections;"
```

### Reset Database

```bash
# Drop all tables
alembic downgrade base

# Recreate tables
alembic upgrade head
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

### Database Connection Error

```bash
# Check PostgreSQL is running
pg_isready

# Start PostgreSQL (macOS)
brew services start postgresql@14

# Start PostgreSQL (Linux)
sudo systemctl start postgresql
```

### Selenium/Chrome Issues

```bash
# Install Chrome (macOS)
brew install --cask google-chrome

# Install Chrome (Ubuntu)
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
```

## Next Steps

1. ✅ Backend API is running locally
2. ⏭️ Deploy to Railway (see deployment guide)
3. ⏭️ Build Next.js frontend
4. ⏭️ Deploy frontend to Vercel

---

# API Reference

Complete REST API reference for all cURLinator endpoints.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication-1)
- [Rate Limiting](#rate-limiting)
- [Response Headers](#response-headers)
- [Error Handling](#error-handling)
- [Endpoints](#endpoints)
  - [Authentication](#authentication-endpoints)
  - [User Management](#user-management-endpoints)
  - [Crawling](#crawl-endpoint-1)
  - [Chat](#chat-endpoint)
  - [Sessions](#session-management-endpoints)
  - [Collections](#collection-management-endpoints)
  - [Collection Sharing](#collection-sharing-endpoints)
  - [Admin - User Management](#admin---user-management-endpoints)
  - [Admin - System Prompts](#admin---system-prompt-endpoints)
  - [User Settings](#user-settings-endpoints)
  - [Health Check](#health-check-endpoint)

---

## Overview

**Base URL:** `http://localhost:8000` (development) or your deployed URL

**API Version:** v1

**Interactive Documentation:**

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## Authentication

Most endpoints require JWT authentication. Include the token in the `Authorization` header:

```bash
Authorization: Bearer <your_access_token>
```

### Obtaining a Token

1. Register a new user: `POST /api/v1/auth/register`
2. Login: `POST /api/v1/auth/login`
3. Use the `access_token` from the response in subsequent requests

### Token Expiration

JWT tokens expire after 7 days. After expiration, you'll need to login again to obtain a new token.

---

## Rate Limiting

The API implements rate limiting to prevent abuse:

| Endpoint Category                      | Rate Limit     | Window            |
| -------------------------------------- | -------------- | ----------------- |
| Auth endpoints (`/register`, `/login`) | 10 requests    | per minute per IP |
| Crawl endpoint (`/crawl`)              | 5 requests     | per hour per IP   |
| Chat endpoint (`/chat`)                | 60 requests    | per minute per IP |
| Admin endpoints                        | 30-60 requests | per minute per IP |
| Collection endpoints                   | No limit       | -                 |

### Rate Limit Headers

Rate-limited endpoints include these headers in responses:

- **`X-RateLimit-Limit`**: Maximum requests allowed in the time window
- **`X-RateLimit-Remaining`**: Requests remaining in current window
- **`X-RateLimit-Reset`**: Unix timestamp when the rate limit resets

### Rate Limit Exceeded Response

HTTP 429 with:

```json
{
  "error": "Rate limit exceeded",
  "message": "Too many requests. Please try again later.",
  "suggestion": "Wait a few minutes before making more requests."
}
```

The response includes a `Retry-After` header indicating seconds to wait.

---

## Response Headers

All responses include:

- **`X-Correlation-ID`**: Unique request identifier for tracing
- **`X-Process-Time`**: Request processing time in seconds
- **`X-RateLimit-*`**: Rate limiting information (on rate-limited endpoints)

---

## Error Handling

### Standard Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common HTTP Status Codes

- **`200 OK`**: Request successful
- **`201 Created`**: Resource created successfully
- **`400 Bad Request`**: Invalid request parameters
- **`401 Unauthorized`**: Missing or invalid authentication token
- **`403 Forbidden`**: Authenticated but not authorized for this resource
- **`404 Not Found`**: Resource not found
- **`422 Unprocessable Entity`**: Validation error
- **`429 Too Many Requests`**: Rate limit exceeded
- **`500 Internal Server Error`**: Server error (check logs)

---

## Endpoints

## Endpoints

### Authentication Endpoints

#### Register a New User

```http
POST /api/v1/auth/register
```

**Description:** Create a new user account.

**Authentication Required:** No

**Rate Limit:** 10 requests/minute per IP

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
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "user@example.com",
    "role": "user",
    "is_active": true,
    "created_at": "2025-11-15T10:30:00Z"
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

- **`400 Bad Request`**: Email already registered
- **`422 Unprocessable Entity`**: Validation error (weak password, invalid email)
- **`429 Too Many Requests`**: Rate limit exceeded
- **`500 Internal Server Error`**: Database error

---

#### Login

```http
POST /api/v1/auth/login
```

**Description:** Authenticate and obtain an access token.

**Authentication Required:** No

**Rate Limit:** 10 requests/minute per IP

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
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "user@example.com",
    "role": "user",
    "is_active": true,
    "created_at": "2025-11-15T10:30:00Z"
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

- **`401 Unauthorized`**: Invalid credentials
- **`422 Unprocessable Entity`**: Invalid email format
- **`429 Too Many Requests`**: Rate limit exceeded
- **`500 Internal Server Error`**: Database error

---

#### Get Current User

```http
GET /api/v1/auth/me
```

**Description:** Get information about the currently authenticated user.

**Authentication Required:** Yes

**Response (200 OK):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "role": "user",
  "is_active": true,
  "created_at": "2025-11-15T10:30:00Z"
}
```

**cURL Example:**

```bash
curl -X GET http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Error Responses:**

- **`401 Unauthorized`**: Missing or invalid token

---

### User Management Endpoints

#### Change Password

```http
POST /api/v1/auth/change-password
```

**Description:** Change the password for the currently authenticated user.

**Authentication Required:** Yes

**Request Body:**

```json
{
  "current_password": "OldPass123",
  "new_password": "NewSecurePass456"
}
```

**Response (200 OK):**

```json
{
  "message": "Password changed successfully"
}
```

**cURL Example:**

```bash
curl -X POST http://localhost:8000/api/v1/auth/change-password \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "current_password": "OldPass123",
    "new_password": "NewSecurePass456"
  }'
```

**Error Responses:**

- **`401 Unauthorized`**: Current password is incorrect
- **`422 Unprocessable Entity`**: New password doesn't meet requirements

---

#### Delete Account

```http
DELETE /api/v1/auth/me
```

**Description:** Permanently delete the currently authenticated user's account and all associated data.

**Authentication Required:** Yes

**Response (200 OK):**

```json
{
  "message": "Account deleted successfully"
}
```

**cURL Example:**

```bash
curl -X DELETE http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Warning:** This action is irreversible. All collections, chat sessions, and user data will be permanently deleted.

---

### Crawl Endpoint

#### Crawl API Documentation

```http
POST /api/v1/crawl
```

**Description:** Scrape and index API documentation from a URL.

**Authentication Required:** Yes

**Rate Limit:** 5 requests/hour per IP

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

- **`url`** (required): Documentation URL to crawl
- **`max_pages`** (optional): Maximum pages to crawl (default: 50, max: 100)
- **`max_depth`** (optional): Maximum crawl depth (default: 3, max: 5)
- **`embedding_provider`** (optional): Embedding model provider
  - `LOCAL`: Local BAAI/bge-small-en-v1.5 (default, free)
  - `OPENAI`: OpenAI text-embedding-3-small (requires API key in user settings)
  - `GEMINI`: Google Gemini embedding (requires API key in user settings)
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

- **`400 Bad Request`**: No documents found or invalid URL
- **`402 Payment Required`**: Premium embedding provider requires API key (BYOK)
- **`403 Forbidden`**: Not authenticated
- **`408 Request Timeout`**: Crawl operation timed out (>10 minutes)
- **`422 Unprocessable Entity`**: Validation error (invalid URL, parameters out of range)
- **`429 Too Many Requests`**: Rate limit exceeded (5/hour)
- **`500 Internal Server Error`**: Crawl failed or embedding model error

---

### Chat Endpoint

#### Query Indexed Documentation

```http
POST /api/v1/chat
```

**Description:** Ask questions about indexed API documentation using natural language.

**Authentication Required:** Yes

**Rate Limit:** 60 requests/minute per IP

**Request Body:**

```json
{
  "collection_name": "crawl_550e8400-e29b-41d4-a716-446655440000",
  "message": "How do I create a new pet? Give me a cURL command.",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Parameters:**

- **`collection_name`** (required): Collection name from crawl response
- **`message`** (required): Natural language query
- **`session_id`** (optional): Chat session ID for conversation history. If not provided, a new session will be created.
- **`conversation_history`** (optional, deprecated): Use `session_id` instead for server-side history management.

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

The chat endpoint supports server-side conversation history management through sessions:

1. **First message**: Send without `session_id`. Response includes a new `session_id`.
2. **Follow-up messages**: Include the `session_id` to maintain conversation context.
3. **Session management**: Use session endpoints to list, view, delete, or reset sessions.

**Multi-turn Conversation Example:**

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

- **`402 Payment Required`**: Free message limit exceeded, requires API key (BYOK)
- **`403 Forbidden`**: Not authenticated or collection not owned by user
- **`404 Not Found`**: Collection or session not found
- **`422 Unprocessable Entity`**: Validation error
- **`429 Too Many Requests`**: Rate limit exceeded (60/minute)
- **`500 Internal Server Error`**: Query failed

---

### Session Management Endpoints

#### List Chat Sessions

```http
GET /api/v1/sessions
```

**Description:** List all chat sessions for the authenticated user, ordered by most recently updated.

**Authentication Required:** Yes

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

```http
GET /api/v1/sessions/{session_id}
```

**Description:** Get detailed information about a specific chat session including all messages.

**Authentication Required:** Yes

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

- **`404 Not Found`**: Session not found or you don't have access to it

---

#### Delete Session

```http
DELETE /api/v1/sessions/{session_id}
```

**Description:** Delete a chat session and all its messages permanently.

**Authentication Required:** Yes

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

- **`404 Not Found`**: Session not found or you don't have access to it

---

#### Reset Session

```http
POST /api/v1/sessions/{session_id}/reset
```

**Description:** Clear all messages from a chat session while keeping the session itself. Useful for starting a fresh conversation on the same collection.

**Authentication Required:** Yes

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

- **`404 Not Found`**: Session not found or you don't have access to it

---

### Collection Management Endpoints

#### List Collections

```http
GET /api/v1/collections
```

**Description:** List all collections owned by or shared with the authenticated user.

**Authentication Required:** Yes

**Response (200 OK):**

```json
{
  "collections": [
    {
      "name": "crawl_550e8400-e29b-41d4-a716-446655440000",
      "base_url": "https://petstore3.swagger.io",
      "pages_crawled": 15,
      "created_at": "2025-11-15T10:30:00Z",
      "visibility": "private"
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

```http
GET /api/v1/collections/{collection_name}
```

**Description:** Get detailed information about a specific collection.

**Authentication Required:** Yes

**Response (200 OK):**

```json
{
  "name": "crawl_550e8400-e29b-41d4-a716-446655440000",
  "base_url": "https://petstore3.swagger.io",
  "pages_crawled": 15,
  "created_at": "2025-11-15T10:30:00Z",
  "embedding_provider": "LOCAL",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "visibility": "private"
}
```

**cURL Example:**

```bash
curl -X GET http://localhost:8000/api/v1/collections/crawl_550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer $TOKEN"
```

**Error Responses:**

- **`403 Forbidden`**: Not authenticated or collection not owned by user
- **`404 Not Found`**: Collection not found

---

#### Delete Collection

```http
DELETE /api/v1/collections/{collection_name}
```

**Description:** Permanently delete a collection and all associated data.

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

- **`403 Forbidden`**: Not authenticated or collection not owned by user
- **`404 Not Found`**: Collection not found
- **`500 Internal Server Error`**: Failed to delete collection

---

### Collection Sharing Endpoints

cURLinator allows you to share collections with other users and control their access permissions. Collections can be private (owner-only), shared with specific users, or made public for everyone.

#### Permission Levels

- **`view`**: User can view collection metadata only (name, URL, page count, etc.)
- **`chat`**: User can view collection metadata AND query the collection via the chat endpoint

#### Visibility Options

- **`private`**: Only the owner can access the collection (default)
- **`public`**: Anyone can view and chat with the collection (no sharing needed)

---

#### Share Collection with User

```http
POST /api/v1/collections/{collection_name}/share
```

**Description:** Share a collection with another user by email address.

**Authentication Required:** Yes (must be collection owner)

**Request Body:**

```json
{
  "user_email": "colleague@example.com",
  "permission": "chat"
}
```

**Parameters:**

- **`user_email`** (required): Email address of the user to share with
- **`permission`** (required): Permission level - `"view"` or `"chat"`

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

- **`403 Forbidden`**: Not authenticated or not the collection owner
- **`404 Not Found`**: Collection or target user not found
- **`400 Bad Request`**: Invalid permission level or trying to share with yourself

---

#### List Collection Shares

```http
GET /api/v1/collections/{collection_name}/shares
```

**Description:** List all users who have access to a collection.

**Authentication Required:** Yes (must be collection owner)

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

- **`403 Forbidden`**: Not authenticated or not the collection owner
- **`404 Not Found`**: Collection not found

---

#### Update Share Permission

```http
PATCH /api/v1/collections/{collection_name}/shares/{user_email}
```

**Description:** Update the permission level for an existing share.

**Authentication Required:** Yes (must be collection owner)

**Request Body:**

```json
{
  "permission": "view"
}
```

**Parameters:**

- **`permission`** (required): New permission level - `"view"` or `"chat"`

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

- **`403 Forbidden`**: Not authenticated or not the collection owner
- **`404 Not Found`**: Collection or share not found

---

#### Revoke Share

```http
DELETE /api/v1/collections/{collection_name}/shares/{user_email}
```

**Description:** Revoke a user's access to a shared collection.

**Authentication Required:** Yes (must be collection owner)

**Response (204 No Content):**

No response body.

**cURL Example:**

```bash
curl -X DELETE http://localhost:8000/api/v1/collections/my_api_docs/shares/colleague@example.com \
  -H "Authorization: Bearer $TOKEN"
```

**Error Responses:**

- **`403 Forbidden`**: Not authenticated or not the collection owner
- **`404 Not Found`**: Collection or share not found

---

#### Update Collection Visibility

```http
PATCH /api/v1/collections/{collection_name}/visibility
```

**Description:** Update the visibility of a collection (private or public).

**Authentication Required:** Yes (must be collection owner)

**Request Body:**

```json
{
  "visibility": "public"
}
```

**Parameters:**

- **`visibility`** (required): Visibility level - `"private"` or `"public"`

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

- **`403 Forbidden`**: Not authenticated or not the collection owner
- **`404 Not Found`**: Collection not found

---

### Admin - User Management Endpoints

These endpoints are only accessible to users with the `admin` role.

#### List All Users

```http
GET /api/v1/admin/users
```

**Description:** List all users in the system (admin only).

**Authentication Required:** Yes (admin only)

**Rate Limit:** 60 requests/minute

**Response (200 OK):**

```json
{
  "users": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "email": "user@example.com",
      "role": "user",
      "is_active": true,
      "created_at": "2025-11-15T10:30:00Z"
    }
  ],
  "total": 1
}
```

**cURL Example:**

```bash
curl -X GET http://localhost:8000/api/v1/admin/users \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

**Error Responses:**

- **`403 Forbidden`**: Not authenticated or not an admin user

---

#### Get User Details

```http
GET /api/v1/admin/users/{user_id}
```

**Description:** Get detailed information about a specific user (admin only).

**Authentication Required:** Yes (admin only)

**Rate Limit:** 60 requests/minute

**Response (200 OK):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "role": "user",
  "is_active": true,
  "created_at": "2025-11-15T10:30:00Z",
  "collections_count": 5,
  "sessions_count": 12
}
```

**cURL Example:**

```bash
curl -X GET http://localhost:8000/api/v1/admin/users/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

**Error Responses:**

- **`403 Forbidden`**: Not authenticated or not an admin user
- **`404 Not Found`**: User not found

---

#### Delete User

```http
DELETE /api/v1/admin/users/{user_id}
```

**Description:** Permanently delete a user and all associated data (admin only).

**Authentication Required:** Yes (admin only)

**Rate Limit:** 30 requests/minute

**Response (200 OK):**

```json
{
  "message": "User deleted successfully",
  "user_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**cURL Example:**

```bash
curl -X DELETE http://localhost:8000/api/v1/admin/users/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

**Error Responses:**

- **`403 Forbidden`**: Not authenticated or not an admin user
- **`404 Not Found`**: User not found

**Warning:** This action is irreversible. All user data, collections, and sessions will be permanently deleted.

---

### Admin - System Prompt Endpoints

These endpoints allow admins to customize the system prompts used by the ChatAgent for A/B testing and experimentation.

#### 3-Tier Prompt Resolution

The system uses a 3-tier fallback mechanism for prompt resolution:

1. **User Custom Prompt**: If a user has a custom prompt assigned (for A/B testing), use it
2. **System-Wide Prompt**: If no user custom prompt, use the system-wide default from database
3. **Hardcoded Default**: If no system-wide prompt is set, use the ChatAgent's hardcoded default

---

#### Set System-Wide Prompt

```http
PATCH /api/v1/admin/system-prompt
```

**Description:** Set the system-wide default prompt for all users (admin only).

**Authentication Required:** Yes (admin only)

**Rate Limit:** 30 requests/minute

**Request Body:**

```json
{
  "prompt": "You are a helpful API documentation assistant. Always provide accurate cURL commands...",
  "description": "Production system prompt v2.0"
}
```

**Parameters:**

- **`prompt`** (required): The system prompt text (max 10,000 characters)
- **`description`** (optional): Description of the prompt purpose

**Response (200 OK):**

```json
{
  "message": "System-wide prompt updated successfully",
  "prompt_preview": "You are a helpful API documentation assistant. Always provide accurate cURL commands...",
  "updated_at": "2025-11-19T10:30:00Z",
  "updated_by": "admin@curlinator.com"
}
```

**cURL Example:**

```bash
curl -X PATCH http://localhost:8000/api/v1/admin/system-prompt \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "You are a helpful API documentation assistant...",
    "description": "Production system prompt v2.0"
  }'
```

**Error Responses:**

- **`403 Forbidden`**: Not authenticated or not an admin user
- **`422 Unprocessable Entity`**: Validation error (prompt too long, empty prompt)

---

#### Get Prompts Overview

```http
GET /api/v1/admin/prompts
```

**Description:** View current system-wide prompt and list users with custom prompts (admin only).

**Authentication Required:** Yes (admin only)

**Rate Limit:** 60 requests/minute

**Response (200 OK):**

```json
{
  "system_prompt": {
    "prompt": "You are a helpful API documentation assistant...",
    "description": "Production system prompt v2.0",
    "updated_at": "2025-11-19T10:30:00Z",
    "updated_by_email": "admin@curlinator.com",
    "is_default": false
  },
  "users_with_custom_prompts": [
    {
      "user_id": "user123...",
      "user_email": "testuser@example.com",
      "variant_name": "variant_b_auth_focus",
      "prompt_preview": "You are an API assistant focused on authentication...",
      "updated_at": "2025-11-19T09:00:00Z"
    }
  ],
  "total_users_with_custom_prompts": 1
}
```

**cURL Example:**

```bash
curl -X GET http://localhost:8000/api/v1/admin/prompts \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

**Error Responses:**

- **`403 Forbidden`**: Not authenticated or not an admin user

---

#### Set Per-User Custom Prompt

```http
PATCH /api/v1/admin/users/{user_id}/prompt
```

**Description:** Set a custom prompt for a specific user for A/B testing (admin only).

**Authentication Required:** Yes (admin only)

**Rate Limit:** 30 requests/minute

**Request Body:**

```json
{
  "prompt": "You are an API assistant focused on authentication and security best practices...",
  "variant_name": "variant_b_auth_focus"
}
```

**Parameters:**

- **`prompt`** (required): The custom prompt text (max 10,000 characters)
- **`variant_name`** (optional): Name/label for this prompt variant (for tracking A/B tests)

**Response (200 OK):**

```json
{
  "message": "Custom prompt set for user testuser@example.com",
  "prompt_preview": "You are an API assistant focused on authentication and security best practices...",
  "updated_at": "2025-11-19T10:30:00Z",
  "updated_by": "admin@curlinator.com"
}
```

**cURL Example:**

```bash
curl -X PATCH http://localhost:8000/api/v1/admin/users/user123.../prompt \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "You are an API assistant focused on authentication...",
    "variant_name": "variant_b_auth_focus"
  }'
```

**Error Responses:**

- **`403 Forbidden`**: Not authenticated or not an admin user
- **`404 Not Found`**: User not found
- **`422 Unprocessable Entity`**: Validation error (prompt too long, empty prompt)

---

#### Reset User Prompt

```http
DELETE /api/v1/admin/users/{user_id}/prompt
```

**Description:** Remove custom prompt for a user, reverting them to system default (admin only).

**Authentication Required:** Yes (admin only)

**Rate Limit:** 30 requests/minute

**Response (200 OK):**

```json
{
  "message": "Custom prompt removed for user testuser@example.com. User will now use system default.",
  "user_id": "user123...",
  "user_email": "testuser@example.com"
}
```

**cURL Example:**

```bash
curl -X DELETE http://localhost:8000/api/v1/admin/users/user123.../prompt \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

**Error Responses:**

- **`403 Forbidden`**: Not authenticated or not an admin user
- **`404 Not Found`**: User not found or user doesn't have a custom prompt

---

### User Settings Endpoints

#### Get User Settings

```http
GET /api/v1/settings
```

**Description:** Get settings for the currently authenticated user.

**Authentication Required:** Yes

**Response (200 OK):**

```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "preferred_embedding_provider": "local",
  "default_max_pages": 50,
  "default_max_depth": 3,
  "openai_api_key_configured": false,
  "gemini_api_key_configured": false,
  "anthropic_api_key_configured": false,
  "free_messages_used": 5,
  "free_messages_limit": 10,
  "last_message_reset_date": "2025-11-15T00:00:00Z"
}
```

**cURL Example:**

```bash
curl -X GET http://localhost:8000/api/v1/settings \
  -H "Authorization: Bearer $TOKEN"
```

---

#### Update User Settings

```http
PATCH /api/v1/settings
```

**Description:** Update settings for the currently authenticated user. This is where users configure their BYOK (Bring Your Own Key) API keys for premium features.

**Authentication Required:** Yes

**Request Body:**

```json
{
  "preferred_embedding_provider": "openai",
  "default_max_pages": 100,
  "default_max_depth": 5,
  "openai_api_key": "sk-...",
  "gemini_api_key": "AIza...",
  "anthropic_api_key": "sk-ant-..."
}
```

**Parameters:**

All parameters are optional. Only include the fields you want to update.

- **`preferred_embedding_provider`**: Preferred embedding provider (`"local"`, `"openai"`, `"gemini"`, `"auto"`)
- **`default_max_pages`**: Default maximum pages to crawl (1-100)
- **`default_max_depth`**: Default maximum crawl depth (1-5)
- **`openai_api_key`**: OpenAI API key (encrypted at rest)
- **`gemini_api_key`**: Google Gemini API key (encrypted at rest)
- **`anthropic_api_key`**: Anthropic API key (encrypted at rest)

**Response (200 OK):**

```json
{
  "message": "Settings updated successfully",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "preferred_embedding_provider": "openai",
  "default_max_pages": 100,
  "default_max_depth": 5,
  "openai_api_key_configured": true,
  "gemini_api_key_configured": true,
  "anthropic_api_key_configured": true
}
```

**cURL Example:**

```bash
curl -X PATCH http://localhost:8000/api/v1/settings \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "openai_api_key": "sk-...",
    "preferred_embedding_provider": "openai"
  }'
```

**BYOK (Bring Your Own Key) - Freemium Model:**

- **Free Tier**: 10 free chat messages per month using local embeddings
- **Premium Features**: Unlimited messages and premium embedding providers (OpenAI, Gemini) by adding your own API keys
- API keys are encrypted at rest using AES-256 encryption
- Keys are never logged or exposed in responses

**Error Responses:**

- **`422 Unprocessable Entity`**: Validation error (invalid provider, parameters out of range)

---

### Health and Metrics Endpoints

#### Health Check

```http
GET /health
```

**Description:** Check API health and system status.

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

- **`healthy`**: All systems operational
- **`degraded`**: System operational but with warnings (high CPU/memory/disk usage or LLM API connectivity issues)
- **`unhealthy`**: Critical system failure (database or Chroma unavailable)

**System Warnings:**

The health check monitors system resources and returns warnings when:

- CPU usage > 90%
- Memory usage > 90%
- Disk usage > 90%

**LLM Connectivity Checks:**

- **OpenAI**: Lists available models to verify API access
- **Gemini**: Lists available models to verify API access
- **Anthropic**: Validates API key configuration (no actual API call to save costs)

These checks are **non-blocking** - LLM API failures will mark the service as `degraded` but not `unhealthy`.

**cURL Example:**

```bash
curl -X GET http://localhost:8000/health
```

---

#### Prometheus Metrics

```http
GET /metrics
```

**Description:** Prometheus-formatted metrics for monitoring.

**Authentication Required:** No

**Response (200 OK):**

Returns Prometheus-formatted metrics in plain text.

**cURL Example:**

```bash
curl -X GET http://localhost:8000/metrics
```

**Note:** For detailed Prometheus configuration and Grafana dashboard setup, see the main [README.md](README.md).

---

## Support

For issues, questions, or feature requests:

- **GitHub Issues**: [https://github.com/kishorekumar1308/curlinator/issues](https://github.com/kishorekumar1308/curlinator/issues)
- **Interactive API Docs**: Visit `/docs` when running the server
- **Main Documentation**: See [README.md](README.md) for setup and deployment guides
