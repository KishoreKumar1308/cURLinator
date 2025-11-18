#!/bin/bash

# cURLinator Local API Test Script

BASE_URL="http://localhost:8000"

echo "🧪 Testing cURLinator Local API"
echo "=========================================="
echo ""

# Test 1: Health Check
echo "Test 1: Health Check"
echo "--------------------"
HEALTH=$(curl -s -X GET "$BASE_URL/health")
echo "$HEALTH" | jq
HEALTH_STATUS=$(echo "$HEALTH" | jq -r '.status')
if [ "$HEALTH_STATUS" = "healthy" ]; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi
echo ""

# Test 2: User Registration
echo "Test 2: User Registration"
echo "-------------------------"
REGISTER_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test-'$(date +%s)'@example.com",
    "password": "SecurePassword123!",
    "full_name": "Test User"
  }')
echo "$REGISTER_RESPONSE" | jq
ACCESS_TOKEN=$(echo "$REGISTER_RESPONSE" | jq -r '.access_token')
if [ -n "$ACCESS_TOKEN" ] && [ "$ACCESS_TOKEN" != "null" ]; then
    echo "✅ User registration passed"
    echo "Access Token: ${ACCESS_TOKEN:0:50}..."
else
    echo "❌ User registration failed"
    exit 1
fi
echo ""

# Test 3: Protected Endpoint
echo "Test 3: Protected Endpoint (/auth/me)"
echo "--------------------------------------"
ME_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/auth/me" \
  -H "Authorization: Bearer $ACCESS_TOKEN")
echo "$ME_RESPONSE" | jq
USER_EMAIL=$(echo "$ME_RESPONSE" | jq -r '.email')
if [ -n "$USER_EMAIL" ] && [ "$USER_EMAIL" != "null" ]; then
    echo "✅ JWT authentication passed"
else
    echo "❌ JWT authentication failed"
    exit 1
fi
echo ""

# Test 4: Crawl API Documentation
echo "Test 4: Crawl API Documentation"
echo "--------------------------------"
echo "⏳ Crawling https://httpbin.org (this may take 10-30 seconds)..."
CRAWL_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/crawl" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://httpbin.org",
    "max_pages": 5,
    "max_depth": 1
  }')
echo "$CRAWL_RESPONSE" | jq
COLLECTION_NAME=$(echo "$CRAWL_RESPONSE" | jq -r '.collection_name')
PAGES_CRAWLED=$(echo "$CRAWL_RESPONSE" | jq -r '.pages_crawled')
if [ -n "$COLLECTION_NAME" ] && [ "$COLLECTION_NAME" != "null" ] && [ "$PAGES_CRAWLED" -gt 0 ]; then
    echo "✅ Crawl functionality passed"
    echo "Collection: $COLLECTION_NAME"
    echo "Pages crawled: $PAGES_CRAWLED"
else
    echo "❌ Crawl functionality failed"
    exit 1
fi
echo ""

# Test 5: Chat with Crawled Documentation
echo "Test 5: Chat with Crawled Documentation"
echo "----------------------------------------"
echo "⏳ Asking question (this may take 5-10 seconds)..."
CHAT_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/chat" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"collection_name\": \"$COLLECTION_NAME\",
    \"message\": \"What endpoints are available?\"
  }")

# Try to parse with jq, but show raw response if jq fails
if echo "$CHAT_RESPONSE" | jq . > /dev/null 2>&1; then
    echo "$CHAT_RESPONSE" | jq .
    CHAT_ANSWER=$(echo "$CHAT_RESPONSE" | jq -r '.response')
else
    echo "⚠️  Response contains special characters, showing raw output:"
    echo "$CHAT_RESPONSE"
    # Try to extract response field manually
    CHAT_ANSWER=$(echo "$CHAT_RESPONSE" | grep -o '"response":"[^"]*"' | sed 's/"response":"\(.*\)"/\1/' | head -1)
fi

if [ -n "$CHAT_ANSWER" ] && [ "$CHAT_ANSWER" != "null" ]; then
    echo "✅ Chat functionality passed"
    echo ""
    echo "Full Answer:"
    echo "----------------------------------------"
    echo "$CHAT_ANSWER"
    echo "----------------------------------------"
else
    echo "❌ Chat functionality failed"
    exit 1
fi
echo ""

# Test 6: List Collections
echo "Test 6: List Collections"
echo "------------------------"
COLLECTIONS_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/collections" \
  -H "Authorization: Bearer $ACCESS_TOKEN")
echo "$COLLECTIONS_RESPONSE" | jq
TOTAL_COLLECTIONS=$(echo "$COLLECTIONS_RESPONSE" | jq '. | length')
if [ "$TOTAL_COLLECTIONS" -gt 0 ] 2>/dev/null; then
    echo "✅ Collections persistence passed"
    echo "Total collections: $TOTAL_COLLECTIONS"
else
    echo "❌ Collections persistence failed"
    exit 1
fi
echo ""

# Summary
echo "=========================================="
echo "🎉 All tests passed!"
echo "=========================================="
echo ""
echo "✅ Health check: Working"
echo "✅ Database: Connected and migrated"
echo "✅ Authentication: Working (JWT)"
echo "✅ Selenium/Chrome: Working"
echo "✅ Gemini LLM: Working"
echo "✅ Chroma Vector Store: Working"
echo "✅ Crawl functionality: Working"
echo "✅ Chat functionality: Working"
echo ""
echo "🚀 Your local API is fully functional!"
echo ""
echo "📝 Next Steps:"
echo "   - Check server logs for BM25 retriever confirmation:"
echo "     Look for: '[ChatAgent] Retrieved X nodes from vector store'"
echo "     Look for: '[ChatAgent] Created BM25 retriever (top_k=5)'"
echo "     Look for: '[ChatAgent] Created hybrid retriever (vector + BM25)'"
echo ""

