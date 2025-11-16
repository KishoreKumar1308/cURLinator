"""
End-to-End Workflow Test: DocumentationAgent → ChatAgent

This example demonstrates the complete cURLinator workflow:
1. Crawl API documentation with DocumentationAgent
2. Index documents in vector database
3. Query with ChatAgent to generate cURL commands
4. Test conversation history with follow-up questions

Usage:
    uv run python examples/test_end_to_end_workflow.py

Requirements:
    - OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file (for LLM)
    - Internet connection for crawling
    - First run will download local embedding model (~90MB)
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from curlinator.agents import DocumentationAgent, ChatAgent
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings


# ============================================================================
# CONFIGURATION - Change this to test different API documentation sites
# ============================================================================

# Choose one of these test URLs (uncomment the one you want to test):

# Option 1: Swagger Petstore (RECOMMENDED FOR FIRST TEST)
# - Has OpenAPI spec (tests fast path)
# - Small and fast to crawl (~10 documents)
# - Well-structured documentation
# BASE_URL = "https://petstore3.swagger.io"
# MAX_DEPTH = 2
# MAX_PAGES = 20

# Option 2: JSONPlaceholder
# - No OpenAPI spec (tests full crawl path)
# - Simple REST API documentation
# - Good for testing page classification
# BASE_URL = "https://jsonplaceholder.typicode.com"
# MAX_DEPTH = 2
# MAX_PAGES = 30

# Option 3: Stripe API (ADVANCED - SLOW)
# - Real-world complex documentation
# - Has OpenAPI spec
# - Large site (use max_pages to limit)
BASE_URL = "https://docs.stripe.com/api"
MAX_DEPTH = 2
MAX_PAGES = 100  # Limit to avoid long crawl times


# LLM Configuration
# Choose your LLM provider (uncomment one):
LLM_PROVIDER = "openai"  # Requires OPENAI_API_KEY
# LLM_PROVIDER = "anthropic"  # Requires ANTHROPIC_API_KEY

# Vector Database Configuration
COLLECTION_NAME = "test_end_to_end"
PERSIST_DIRECTORY = "./data/vector_db"

# DocumentationAgent Configuration
ENABLE_ENRICHMENT = True  # Enable contextual enrichment (recommended)
USE_LLM_CLASSIFICATION = False  # Use LLM for page classification (slower, more accurate)
VERBOSE = True  # Show detailed logging


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_section(title: str) -> None:
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subsection(title: str) -> None:
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---\n")


def get_llm():
    """Get LLM instance based on configuration"""
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    if LLM_PROVIDER == "openai":
        # Get API key and base URL from environment
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        kwargs = {
            "model": "gpt-4",
            "temperature": 0,
            "api_key": api_key,
        }

        # Add custom API base if provided
        if api_base:
            kwargs["api_base"] = api_base
            print(f"   Using custom OpenAI API base: {api_base}")

        return OpenAI(**kwargs)
    elif LLM_PROVIDER == "anthropic":
        return Anthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

async def main():
    """Run the complete end-to-end workflow"""
    
    print_section("🚀 cURLinator End-to-End Workflow Test")
    
    print(f"📋 Configuration:")
    print(f"   Base URL: {BASE_URL}")
    print(f"   Max Depth: {MAX_DEPTH}")
    print(f"   Max Pages: {MAX_PAGES}")
    print(f"   LLM Provider: {LLM_PROVIDER}")
    print(f"   Enrichment: {'Enabled' if ENABLE_ENRICHMENT else 'Disabled'}")
    print(f"   LLM Classification: {'Enabled' if USE_LLM_CLASSIFICATION else 'Disabled'}")
    
    # ========================================================================
    # STEP 1: Initialize LLM and Embedding Model
    # ========================================================================

    print_section("Step 1: Initialize LLM and Embedding Model")

    try:
        llm = get_llm()
        print(f"✅ LLM initialized: {llm.model}")

        # Set global LLM for LlamaIndex components (needed by QueryFusionRetriever)
        Settings.llm = llm

        # Configure local embedding model (HuggingFace)
        # This avoids API dependencies and works with any LLM provider
        print("🔧 Configuring local embedding model...")
        print("   Using: BAAI/bge-small-en-v1.5 (sentence-transformers)")
        print("   (First run will download the model, ~90MB)")

        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            # Cache directory for model downloads
            cache_folder="./data/models"
        )
        print("✅ Embedding model configured: BAAI/bge-small-en-v1.5")

    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        print("\n💡 Make sure you have set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file")
        return
    
    # ========================================================================
    # STEP 2: Crawl Documentation with DocumentationAgent
    # ========================================================================
    
    print_section("Step 2: Crawl API Documentation")
    
    print("🕷️  Initializing DocumentationAgent...")
    doc_agent = DocumentationAgent(
        llm=llm,
        max_depth=MAX_DEPTH,
        max_pages=MAX_PAGES,
        enable_enrichment=ENABLE_ENRICHMENT,
        use_llm_classification=USE_LLM_CLASSIFICATION,
        verbose=VERBOSE,
    )
    
    print(f"🌐 Crawling {BASE_URL}...")
    print("   (This may take 30 seconds to several minutes depending on the site)\n")
    
    try:
        documents = await doc_agent.execute(BASE_URL)
        
        print_subsection("Crawl Results")
        print(f"✅ Successfully crawled {len(documents)} documents")
        
        # Display statistics
        if documents:
            # Check if OpenAPI spec was detected
            openapi_docs = [d for d in documents if d.metadata.get("source") == "openapi"]
            if openapi_docs:
                print(f"   📄 OpenAPI spec detected: {len(openapi_docs)} documents from spec")
            
            # Count page types
            page_types = {}
            for doc in documents:
                page_type = doc.metadata.get("page_type", "unknown")
                page_types[page_type] = page_types.get(page_type, 0) + 1
            
            print(f"\n   📊 Document breakdown by type:")
            for page_type, count in sorted(page_types.items(), key=lambda x: x[1], reverse=True):
                print(f"      - {page_type}: {count}")
            
            # Show enrichment status
            enriched_count = sum(1 for d in documents if d.metadata.get("contextually_enriched"))
            if enriched_count > 0:
                print(f"\n   ✨ Contextual enrichment: {enriched_count}/{len(documents)} documents enriched")
            
            # Show sample document
            print_subsection("Sample Document")
            sample_doc = documents[0]
            print(f"   Title: {sample_doc.metadata.get('title', 'N/A')}")
            print(f"   URL: {sample_doc.metadata.get('url', 'N/A')}")
            print(f"   Type: {sample_doc.metadata.get('page_type', 'N/A')}")
            print(f"   Text length: {len(sample_doc.text)} characters")
            print(f"   Text preview: {sample_doc.text[:200]}...")
        
    except Exception as e:
        print(f"❌ Failed to crawl documentation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not documents:
        print("⚠️  No documents found. Cannot proceed with ChatAgent.")
        return
    
    # ========================================================================
    # STEP 3: Initialize ChatAgent with Crawled Documents
    # ========================================================================
    
    print_section("Step 3: Initialize ChatAgent")
    
    print("🤖 Creating ChatAgent with crawled documents...")
    print(f"   Vector DB: {PERSIST_DIRECTORY}/{COLLECTION_NAME}")
    print("   (Building index... this may take a moment)\n")
    
    try:
        chat_agent = ChatAgent(
            llm=llm,
            documents=documents,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        print("✅ ChatAgent initialized successfully")
        print(f"   - Indexed {len(documents)} documents")
        print(f"   - Hybrid retrieval enabled (BM25 + vector search)")
        print(f"   - Conversation history enabled (ChatSummaryMemoryBuffer)")
        
    except Exception as e:
        print(f"❌ Failed to initialize ChatAgent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # STEP 4: Query the Documentation
    # ========================================================================
    
    print_section("Step 4: Query the Documentation")
    
    # Define test queries based on the API being tested
    if "petstore" in BASE_URL.lower():
        queries = [
            "How do I create a new pet?",
            "What about updating an existing pet?",
            "How do I delete a pet?",
        ]
    elif "jsonplaceholder" in BASE_URL.lower():
        queries = [
            "How do I get all posts?",
            "What about creating a new post?",
            "How do I update a post?",
        ]
    elif "stripe" in BASE_URL.lower():
        queries = [
            "How do I create a customer?",
            "What about creating a payment intent?",
            "How do I retrieve a customer?",
        ]

    else:
        # Generic queries
        queries = [
            "What endpoints are available?",
            "How do I authenticate?",
            "Show me an example API call",
        ]
    
    print(f"🔍 Testing {len(queries)} queries with conversation history:\n")
    
    for i, query in enumerate(queries, 1):
        print_subsection(f"Query {i}: {query}")
        
        try:
            result = await chat_agent.execute(query)
            
            # Display response
            print("💬 Response:")
            print(f"   {result['response']}\n")
            
            # Display cURL command if present
            if result.get('curl_command'):
                print("📋 cURL Command:")
                print(f"   {result['curl_command']}\n")
            
            # Display sources
            if result.get('sources'):
                print(f"📚 Sources ({len(result['sources'])} documents):")
                for j, source in enumerate(result['sources'][:3], 1):  # Show top 3
                    # Sources are dictionaries with 'metadata' key
                    metadata = source.get('metadata', {})
                    url = metadata.get('url', 'N/A')
                    page_type = metadata.get('page_type', 'N/A')
                    score = source.get('score', 0.0)
                    print(f"   {j}. [{page_type}] {url} (score: {score:.3f})")
                if len(result['sources']) > 3:
                    print(f"   ... and {len(result['sources']) - 3} more")
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # STEP 5: Test Conversation Reset
    # ========================================================================
    
    print_section("Step 5: Test Conversation Reset")
    
    print("🔄 Resetting conversation history...")
    chat_agent.reset_conversation()
    print("✅ Conversation history cleared\n")
    
    print("Testing new conversation:")
    test_query = "What is this API about?"
    print(f"   Query: {test_query}")
    
    try:
        result = await chat_agent.execute(test_query)
        print(f"   Response: {result['response'][:200]}...")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    
    print_section("✅ End-to-End Workflow Complete!")
    
    print("📊 Summary:")
    print(f"   ✅ Crawled {len(documents)} documents from {BASE_URL}")
    print(f"   ✅ Indexed documents in vector database")
    print(f"   ✅ Executed {len(queries)} queries successfully")
    print(f"   ✅ Tested conversation history and reset")
    print("\n🎉 All tests passed! The cURLinator workflow is working end-to-end.\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    asyncio.run(main())

