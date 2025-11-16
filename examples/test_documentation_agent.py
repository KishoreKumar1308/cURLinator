"""
Example script to test Documentation Agent with real APIs.

This demonstrates how to use the Documentation Agent to crawl
and extract API documentation.

The new DocumentationAgent returns List[Document] (LlamaIndex documents)
instead of DocumentationSource. Each Document has:
- text: The document content (with optional contextual enrichment)
- metadata: Dict with url, title, page_type, source, etc.
"""

import asyncio
import json
from curlinator.agents import DocumentationAgent
from curlinator.config import get_settings


def print_document_summary(documents, title="RESULTS"):
    """Helper function to print summary of documents"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"✅ Crawled {len(documents)} documents")

    # Count documents by page type
    page_types = {}
    for doc in documents:
        page_type = doc.metadata.get('page_type', 'unknown')
        page_types[page_type] = page_types.get(page_type, 0) + 1

    print("\n📊 Document Types:")
    for page_type, count in sorted(page_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {page_type}: {count}")

    # Show sample documents
    print("\n📄 Sample Documents:")
    for i, doc in enumerate(documents[:3], 1):
        title = doc.metadata.get('title', 'Untitled')
        url = doc.metadata.get('url', 'N/A')
        page_type = doc.metadata.get('page_type', 'unknown')
        source = doc.metadata.get('source', 'unknown')

        print(f"\n{i}. {title}")
        print(f"   URL: {url}")
        print(f"   Type: {page_type}")
        print(f"   Source: {source}")
        print(f"   Text preview: {doc.text[:100]}...")


async def test_with_simple_api():
    """Test with JSONPlaceholder - simple, well-documented API (no OpenAPI spec)"""
    print("=" * 80)
    print("Testing with JSONPlaceholder API (No OpenAPI spec)")
    print("=" * 80)

    # Initialize agent with new API
    # Note: No cleanup() needed - agent doesn't hold resources
    agent = DocumentationAgent(
        verbose=True,           # Show detailed logging
        max_depth=2,            # Maximum crawl depth from base URL
        max_pages=5,            # Limit to 5 pages for testing
        enable_enrichment=True, # Add contextual prefixes (recommended)
        headless=True,          # Run browser in headless mode
    )

    # Execute - returns List[Document]
    documents = await agent.execute("https://jsonplaceholder.typicode.com")

    # Print results
    print_document_summary(documents, "JSONPlaceholder Results")

    # Check if any documents came from OpenAPI spec
    openapi_docs = [doc for doc in documents if doc.metadata.get('source') == 'openapi']
    if openapi_docs:
        print(f"\n📄 OpenAPI Documents: {len(openapi_docs)}")
    else:
        print(f"\n📄 No OpenAPI spec found (as expected)")

    # Show enrichment status
    enriched_docs = [doc for doc in documents if doc.metadata.get('contextually_enriched')]
    print(f"\n✨ Enriched Documents: {len(enriched_docs)}/{len(documents)}")

    return documents


async def test_with_openapi_api():
    """Test with Swagger Petstore - has OpenAPI spec (fast path)"""
    print("\n" + "=" * 80)
    print("Testing with Swagger Petstore (Has OpenAPI 3.0)")
    print("=" * 80)

    # Initialize agent
    agent = DocumentationAgent(
        verbose=True,
        max_depth=2,
        max_pages=20,           # Won't be used if OpenAPI spec is found
        enable_enrichment=True,
        headless=True,
    )

    # Execute - should detect OpenAPI spec and return early (fast path)
    documents = await agent.execute("https://petstore3.swagger.io")

    # Print results
    print_document_summary(documents, "Swagger Petstore Results")

    # Check if documents came from OpenAPI spec
    openapi_docs = [doc for doc in documents if doc.metadata.get('source') == 'openapi']
    if openapi_docs:
        print(f"\n📄 OpenAPI Spec Detected! (Fast Path)")
        print(f"  - OpenAPI Documents: {len(openapi_docs)}")

        # Show OpenAPI-specific metadata
        for doc in openapi_docs[:3]:
            doc_type = doc.metadata.get('type', 'unknown')
            print(f"\n  Document Type: {doc_type}")
            if doc_type == 'api_overview':
                print(f"    Title: {doc.metadata.get('title', 'N/A')}")
                print(f"    Version: {doc.metadata.get('version', 'N/A')}")
                print(f"    Base URL: {doc.metadata.get('base_url', 'N/A')}")
            elif doc_type == 'api_endpoint':
                print(f"    Method: {doc.metadata.get('method', 'N/A')}")
                print(f"    Endpoint: {doc.metadata.get('endpoint', 'N/A')}")
                print(f"    Tags: {doc.metadata.get('tags', 'N/A')}")
    else:
        print(f"\n⚠️ No OpenAPI spec found (unexpected for Petstore)")

    # Save documents to file
    output_file = "petstore_docs.json"
    with open(output_file, "w") as f:
        # Convert documents to serializable format
        docs_data = [
            {
                "text": doc.text,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]
        json.dump(docs_data, f, indent=2)
    print(f"\n💾 Saved {len(documents)} documents to: {output_file}")

    return documents


async def test_custom_api():
    """Test with your custom API URL"""
    print("\n" + "=" * 80)
    print("Testing with Custom API")
    print("=" * 80)

    # Get URL from user or use default
    api_url = input("Enter API documentation URL (or press Enter for Stripe API): ").strip()
    if not api_url:
        api_url = "https://stripe.com/docs/api"

    # Initialize agent with reasonable limits
    agent = DocumentationAgent(
        verbose=True,
        max_depth=2,
        max_pages=15,           # Limit pages to avoid long crawls
        enable_enrichment=True,
        headless=True,
    )

    # Execute
    documents = await agent.execute(api_url)

    # Print detailed results
    print_document_summary(documents, f"Results for {api_url}")

    # Show metadata statistics
    print("\n📊 Metadata Statistics:")

    # Count unique URLs
    unique_urls = set(doc.metadata.get('url', '') for doc in documents)
    print(f"  - Unique URLs: {len(unique_urls)}")

    # Count documents by source
    sources = {}
    for doc in documents:
        source = doc.metadata.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    print(f"  - Sources: {sources}")

    # Show enrichment status
    enriched_count = sum(1 for doc in documents if doc.metadata.get('contextually_enriched'))
    print(f"  - Enriched: {enriched_count}/{len(documents)}")

    # Show sample metadata
    if documents:
        print("\n📋 Sample Document Metadata:")
        sample_doc = documents[0]
        for key, value in list(sample_doc.metadata.items())[:10]:
            # Truncate long values
            value_str = str(value)
            if len(value_str) > 60:
                value_str = value_str[:60] + "..."
            print(f"  - {key}: {value_str}")

    # Save results
    output_file = f"docs_{api_url.replace('https://', '').replace('/', '_')[:30]}.json"
    with open(output_file, "w") as f:
        docs_data = [
            {
                "text": doc.text,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]
        json.dump(docs_data, f, indent=2)
    print(f"\n💾 Saved {len(documents)} documents to: {output_file}")

    return documents


async def main():
    """Main test function"""
    # Check settings
    settings = get_settings()
    print("🔧 Configuration:")
    print(f"  - LLM Provider: {settings.default_llm_provider}")
    print(f"  - Model: {settings.default_model}")
    print(f"  - Has API Key: {'✅' if settings.openai_api_key or settings.anthropic_api_key or settings.gemini_api_key else '❌'}")

    if not (settings.openai_api_key or settings.anthropic_api_key or settings.gemini_api_key):
        print("\n⚠️  WARNING: No API key found!")
        print("LLM-based page classification will be disabled (rule-based only)")
        print("To enable LLM classification, set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY in .env file")
        print("\nNote: OpenAPI detection and crawling will still work without an API key.")
        print()

    print("\n" + "=" * 80)
    print("DocumentationAgent Test Suite")
    print("=" * 80)
    print("\nChoose test:")
    print("1. JSONPlaceholder (Simple API, no OpenAPI spec - tests full crawl)")
    print("2. Swagger Petstore (Has OpenAPI 3.0 - tests fast path)")
    print("3. Custom URL (Enter your own API documentation URL)")
    print("4. Run all tests (JSONPlaceholder + Petstore)")

    choice = input("\nEnter choice (1-4): ").strip()

    try:
        if choice == "1":
            await test_with_simple_api()
        elif choice == "2":
            await test_with_openapi_api()
        elif choice == "3":
            await test_custom_api()
        elif choice == "4":
            await test_with_simple_api()
            await test_with_openapi_api()
        else:
            print("Invalid choice")
            return

        print("\n" + "=" * 80)
        print("✅ Test completed successfully!")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Test failed with error:")
        print("=" * 80)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

