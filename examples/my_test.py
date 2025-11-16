"""
Your custom test script.

Edit this to test with your preferred API.

This is a minimal example showing how to use the new DocumentationAgent API.
The agent returns List[Document] (LlamaIndex documents) with metadata.
"""

import asyncio
import json
from curlinator.agents import DocumentationAgent


async def main():
    """
    Simple example of using DocumentationAgent.

    The agent will:
    1. Try to detect OpenAPI spec (fast path)
    2. If no spec, crawl the site with WholeSiteReader
    3. Classify page types (api_reference, guide, tutorial, etc.)
    4. Enrich documents with contextual prefixes (optional)
    5. Return List[Document] for use with ChatAgent
    """

    # ========================================================================
    # CONFIGURATION - Edit these values
    # ========================================================================

    # YOUR API URL HERE - Try these examples:
    # - "https://petstore3.swagger.io" (has OpenAPI spec - fast!)
    # - "https://jsonplaceholder.typicode.com" (no spec - full crawl)
    # - "https://stripe.com/docs/api" (has OpenAPI spec)
    api_url = "https://petstore3.swagger.io"

    # Agent configuration
    max_depth = 2           # Maximum crawl depth from base URL
    max_pages = 20          # Maximum pages to crawl (ignored if OpenAPI found)
    enable_enrichment = True  # Add contextual prefixes (recommended)
    verbose = True          # Show detailed logging

    # ========================================================================
    # EXECUTION
    # ========================================================================

    print("=" * 70)
    print(f"🚀 Starting crawl of: {api_url}")
    print("=" * 70)

    # Initialize agent with new API
    # Note: No cleanup() needed - agent doesn't hold resources
    agent = DocumentationAgent(
        verbose=verbose,
        max_depth=max_depth,
        max_pages=max_pages,
        enable_enrichment=enable_enrichment,
        headless=True,  # Run browser in headless mode
    )

    try:
        # Execute - returns List[Document]
        documents = await agent.execute(api_url)

        # ====================================================================
        # RESULTS
        # ====================================================================

        print("\n" + "=" * 70)
        print("✅ COMPLETE!")
        print("=" * 70)
        print(f"Crawled {len(documents)} documents")

        # Count documents by page type
        page_types = {}
        for doc in documents:
            page_type = doc.metadata.get('page_type', 'unknown')
            page_types[page_type] = page_types.get(page_type, 0) + 1

        print("\n📊 Document Types:")
        for page_type, count in sorted(page_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {page_type}: {count}")

        # Check if OpenAPI spec was found
        openapi_docs = [doc for doc in documents if doc.metadata.get('source') == 'openapi']
        if openapi_docs:
            print(f"\n📄 OpenAPI Spec Found! (Fast Path)")
            print(f"  - OpenAPI documents: {len(openapi_docs)}")

            # Show OpenAPI metadata from first document
            if openapi_docs:
                first_doc = openapi_docs[0]
                if first_doc.metadata.get('type') == 'api_overview':
                    print(f"  - Title: {first_doc.metadata.get('title', 'N/A')}")
                    print(f"  - Version: {first_doc.metadata.get('version', 'N/A')}")
                    print(f"  - Base URL: {first_doc.metadata.get('base_url', 'N/A')}")
        else:
            print(f"\n📄 No OpenAPI spec found - used full crawl")

        # Show sample documents
        print(f"\n📄 Sample Documents (showing first 3):")
        for i, doc in enumerate(documents[:3], 1):
            title = doc.metadata.get('title', 'Untitled')
            url = doc.metadata.get('url', 'N/A')
            page_type = doc.metadata.get('page_type', 'unknown')

            print(f"\n{i}. {title}")
            print(f"   URL: {url}")
            print(f"   Type: {page_type}")
            print(f"   Text preview: {doc.text[:80]}...")

        # Show enrichment status
        enriched_count = sum(1 for doc in documents if doc.metadata.get('contextually_enriched'))
        print(f"\n✨ Enriched Documents: {enriched_count}/{len(documents)}")

        # ====================================================================
        # SAVE RESULTS
        # ====================================================================

        output_file = "my_test_results.json"
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

        # ====================================================================
        # NEXT STEPS
        # ====================================================================

        print("\n" + "=" * 70)
        print("🎯 Next Steps:")
        print("=" * 70)
        print("1. Inspect the saved JSON file to see document structure")
        print("2. Use these documents with ChatAgent for RAG-based queries")
        print("3. Try different API URLs to test OpenAPI detection vs crawling")
        print("\nExample ChatAgent usage:")
        print("  from curlinator.agents import ChatAgent")
        print("  chat_agent = ChatAgent()")
        print("  response = await chat_agent.query(documents, 'How do I create a pet?')")

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ Error:")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

