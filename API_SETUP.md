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

## API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

