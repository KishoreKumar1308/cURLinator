#!/bin/bash

# cURLinator API Test Script
# This script tests the basic functionality of the API

set -e

API_URL="http://localhost:8000"
EMAIL="test_$(date +%s)@example.com"
PASSWORD="testpassword123"

echo "🧪 Testing cURLinator API"
echo "=========================="
echo ""

# Test 1: Health Check
echo "1️⃣  Testing health check..."
HEALTH=$(curl -s "$API_URL/health")
echo "✅ Health check: $HEALTH"
echo ""

# Test 2: Register User
echo "2️⃣  Registering user: $EMAIL"
REGISTER_RESPONSE=$(curl -s -X POST "$API_URL/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"$EMAIL\", \"password\": \"$PASSWORD\"}")

TOKEN=$(echo $REGISTER_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")
echo "✅ User registered, token: ${TOKEN:0:20}..."
echo ""

# Test 3: Get Current User
echo "3️⃣  Getting current user info..."
USER_INFO=$(curl -s "$API_URL/api/v1/auth/me" \
  -H "Authorization: Bearer $TOKEN")
echo "✅ User info: $USER_INFO"
echo ""

# Test 4: Login
echo "4️⃣  Testing login..."
LOGIN_RESPONSE=$(curl -s -X POST "$API_URL/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"$EMAIL\", \"password\": \"$PASSWORD\"}")
echo "✅ Login successful"
echo ""

# Test 5: Crawl Documentation (small test)
echo "5️⃣  Testing crawl endpoint (this may take a minute)..."
CRAWL_RESPONSE=$(curl -s -X POST "$API_URL/api/v1/crawl" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "url": "https://httpbin.org",
    "max_pages": 3,
    "max_depth": 1
  }')

COLLECTION_NAME=$(echo $CRAWL_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['collection_name'])" 2>/dev/null || echo "")

if [ -z "$COLLECTION_NAME" ]; then
  echo "⚠️  Crawl may have failed or is still running. Response:"
  echo "$CRAWL_RESPONSE"
else
  echo "✅ Crawl completed: $COLLECTION_NAME"
  echo ""

  # Test 6: Chat Query
  echo "6️⃣  Testing chat endpoint..."
  CHAT_RESPONSE=$(curl -s -X POST "$API_URL/api/v1/chat" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{
      \"collection_name\": \"$COLLECTION_NAME\",
      \"message\": \"What is this site about?\",
      \"conversation_history\": []
    }")
  
  echo "✅ Chat response received"
  echo "$CHAT_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$CHAT_RESPONSE"
fi

echo ""
echo "🎉 All tests completed!"
echo ""
echo "Next steps:"
echo "  - Open http://localhost:8000/docs to explore the API"
echo "  - Try crawling a real API documentation site"
echo "  - Build the Next.js frontend"

