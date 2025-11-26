"""
Agent 2: Chat & Request Generation

This agent interprets user intent and builds API requests
through conversational interaction using RAG (Retrieval Augmented Generation)
with persistent Chroma vector storage and hybrid retrieval (BM25 + vector search).
"""

import logging
import re
from pathlib import Path
from typing import Any

import chromadb
from llama_index.core import QueryBundle, StorageContext, VectorStoreIndex
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from Stemmer import Stemmer

from curlinator.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class ChatAgent(BaseAgent):
    """
    Agent for conversational API interaction and request generation.

    Uses RAG (Retrieval Augmented Generation) with:
    - Persistent Chroma vector storage (survives restarts)
    - Hybrid retrieval (BM25 + vector search for better accuracy)
    - Query fusion for combining retrieval strategies
    - Conversation history management with ChatSummaryMemoryBuffer
      (automatically summarizes old messages instead of truncating)

    Responsibilities:
    1. Index API documentation in persistent vector database
    2. Retrieve relevant endpoints using hybrid search
    3. Maintain conversation context across multiple queries
    4. Resolve user intent to specific API calls
    5. Extract parameters from user messages
    6. Generate cURL commands and request forms
    """

    def __init__(
        self,
        documents: list[Document] | None = None,
        collection_name: str | None = None,
        persist_directory: str | None = None,
        system_prompt: str | None = None,
        llm: Any | None = None,
        embed_model: Any | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize chat agent with documents or load existing index.

        Supports two modes:
        1. **Build mode**: Provide documents to create new index
        2. **Load mode**: No documents, loads existing index from disk

        Args:
            documents: Optional list of LlamaIndex Documents to index
            collection_name: Optional collection name (default from settings)
            persist_directory: Optional directory for persistent storage (default from settings)
            system_prompt: Optional custom system prompt (uses default if not provided)
            llm: Optional LLM instance (default from settings)
            embed_model: Optional embedding model instance (uses global Settings if not provided)
            verbose: Whether to enable verbose logging

        Example:
            >>> # Build mode: Create new index
            >>> agent = ChatAgent(documents=docs, collection_name="stripe_api")

            >>> # Load mode: Load existing index
            >>> agent = ChatAgent(collection_name="stripe_api")

            >>> # With custom embedding model
            >>> from llama_index.embeddings.openai import OpenAIEmbedding
            >>> embed_model = OpenAIEmbedding(model="text-embedding-3-small")
            >>> agent = ChatAgent(
            ...     documents=docs,
            ...     collection_name="stripe_api",
            ...     embed_model=embed_model
            ... )
        """
        # Initialize BaseAgent with only the parameters it accepts
        super().__init__(llm=llm, verbose=verbose)

        # Store ChatAgent-specific parameters
        self.documents = documents or []
        self.collection_name = collection_name or self.settings.vector_collection_name
        self.persist_directory = persist_directory or self.settings.vector_db_path
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.embed_model = embed_model  # Store embedding model (can be None)
        self.index = None
        self.chat_engine = None
        self.chroma_client = None
        self.chroma_collection = None

        # Build or load index
        if documents:
            self._log(f"Building new index with {len(documents)} documents")
            self._build_index()
        else:
            self._log(f"Loading existing index from collection: {self.collection_name}")
            self._load_index()

    def _get_default_system_prompt(self) -> str:
        """
        Get default system prompt optimized for cURL generation.

        Returns:
            Comprehensive system prompt for API documentation assistance
        """
        return """You are cURLinator, an expert API documentation assistant.

CORE RESPONSIBILITIES:
1. Answer questions about API endpoints, authentication, and usage
2. Generate accurate, executable cURL commands
3. Explain API concepts clearly and concisely

CURL COMMAND GUIDELINES:
- Always wrap cURL commands in ```bash code blocks
- Use the exact endpoint URLs from the documentation
- Include authentication (API keys, tokens, OAuth) when mentioned
- Use proper HTTP methods: GET, POST, PUT, PATCH, DELETE
- Format multi-line commands with backslashes (\\) for readability
- Include Content-Type headers for POST/PUT requests
- Use -d for JSON data, -H for headers, -X for methods
- Provide realistic example values for parameters

EXAMPLE FORMAT:
```bash
curl https://api.example.com/v1/resource \\
  -X POST \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "example",
    "value": 123
  }'
```

IMPORTANT RULES:
- Base all answers on the provided documentation context
- If information is missing, state clearly what's not documented
- Don't make assumptions about authentication or parameters
- Explain what each cURL flag does when helpful
- Suggest best practices (error handling, rate limits, etc.)

Now, answer the user's question using the provided API documentation."""

    def _build_index(self) -> None:
        """
        Build and persist vector index with hybrid retrieval.

        Creates:
        1. Chroma persistent client (auto-saves to disk)
        2. Vector store index with embeddings
        3. Hybrid retriever (BM25 + vector search)
        4. Chat engine with conversation memory
        """
        try:
            # 1. Create persistent Chroma client
            db_path = Path(self.persist_directory)
            db_path.mkdir(parents=True, exist_ok=True)

            self.chroma_client = chromadb.PersistentClient(path=str(db_path))
            self._log(f"Created Chroma client at: {db_path}")

            # 2. Get or create collection
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            self._log(f"Using collection: {self.collection_name}")

            # 3. Create vector store and storage context
            vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # 4. Build vector index (automatically persisted by Chroma)
            self._log("Building vector index...")
            index_kwargs = {
                "documents": self.documents,
                "storage_context": storage_context,
                "show_progress": self.verbose,
            }
            # Pass embedding model if provided (otherwise uses global Settings.embed_model)
            if self.embed_model is not None:
                index_kwargs["embed_model"] = self.embed_model
                self._log(f"Using custom embedding model: {type(self.embed_model).__name__}")

            self.index = VectorStoreIndex.from_documents(**index_kwargs)
            self._log(f"✅ Indexed {len(self.documents)} documents")

            # 5. Create chat engine with hybrid retrieval and conversation memory
            self._create_chat_engine()

        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise

    def _load_index(self) -> None:
        """
        Load existing index from persistent Chroma storage.

        Raises:
            ValueError: If collection doesn't exist
        """
        try:
            # 1. Create persistent Chroma client
            db_path = Path(self.persist_directory)
            if not db_path.exists():
                raise ValueError(
                    f"Vector database not found at {db_path}. "
                    "Create an index first by providing documents."
                )

            self.chroma_client = chromadb.PersistentClient(path=str(db_path))
            self._log(f"Connected to Chroma at: {db_path}")

            # 2. Get existing collection
            try:
                self.chroma_collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                self._log(f"Loaded collection: {self.collection_name}")
            except Exception:
                raise ValueError(
                    f"Collection '{self.collection_name}' not found. "
                    "Available collections: "
                    f"{[c.name for c in self.chroma_client.list_collections()]}"
                )

            # 3. Create vector store from existing collection
            vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # 4. Load index from vector store
            self._log("Loading vector index from storage...")
            index_kwargs = {
                "vector_store": vector_store,
                "storage_context": storage_context,
            }
            # Pass embedding model if provided (otherwise uses global Settings.embed_model)
            if self.embed_model is not None:
                index_kwargs["embed_model"] = self.embed_model
                self._log(f"Using custom embedding model: {type(self.embed_model).__name__}")

            self.index = VectorStoreIndex.from_vector_store(**index_kwargs)
            self._log("✅ Index loaded successfully")

            # 5. Create chat engine with conversation memory
            self._create_chat_engine()

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    def _create_chat_engine(self) -> None:
        """
        Create hybrid retriever and chat engine with conversation memory.

        Combines:
        1. Vector retriever (semantic search)
        2. BM25 retriever (lexical search)
        3. Query fusion (reciprocal reranking)
        4. Chat summary memory buffer (conversation history with summarization)
        5. System prompt (cURL generation optimization)

        The ChatSummaryMemoryBuffer automatically summarizes old messages when
        the token limit is reached, preserving important context instead of
        simply truncating the conversation history.

        Raises:
            ValueError: If index is not initialized or LLM is not available
        """
        if not self.index:
            raise ValueError("Index not initialized. Call _build_index() or _load_index() first.")

        if not self.llm:
            raise ValueError(
                "LLM not initialized. Cannot create chat engine without a valid LLM. "
                "Please set a valid API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY) "
                "with a real API key (not a test placeholder)."
            )

        try:
            # 1. Create vector retriever
            vector_retriever = self.index.as_retriever(similarity_top_k=5, verbose=self.verbose)
            self._log("Created vector retriever (top_k=5)")

            # 2. Create BM25 retriever (if possible)
            # Try to get nodes from multiple sources
            nodes = None
            bm25_retriever = None

            # Try 1: Get from docstore (available when building new index)
            if hasattr(self.index, "docstore") and self.index.docstore.docs:
                nodes = list(self.index.docstore.docs.values())
                self._log(f"Got {len(nodes)} nodes from docstore")
            # Try 2: Get from vector store (available when loading from storage)
            elif hasattr(self.index, "vector_store"):
                try:
                    # Retrieve all nodes from vector store using a broad query
                    retriever = self.index.as_retriever(similarity_top_k=100)
                    # Get all nodes by querying with empty string or broad term
                    query_bundle = QueryBundle(query_str="")
                    retrieved_nodes = retriever.retrieve(query_bundle)
                    if retrieved_nodes:
                        nodes = [node.node for node in retrieved_nodes]
                        self._log(f"Retrieved {len(nodes)} nodes from vector store")
                except Exception as e:
                    self._log(f"⚠️  Could not retrieve nodes from vector store: {e}")
            # Try 3: Create from documents (fallback)
            if not nodes and self.documents:
                parser = SentenceSplitter()
                nodes = parser.get_nodes_from_documents(self.documents)
                self._log(f"Created {len(nodes)} nodes from documents")

            # Create BM25 retriever if we have nodes
            if nodes and len(nodes) > 0:
                try:
                    bm25_retriever = BM25Retriever.from_defaults(
                        nodes=nodes,
                        similarity_top_k=5,
                        stemmer=Stemmer("english"),
                        language="english",
                    )
                    self._log("Created BM25 retriever (top_k=5)")
                except Exception as e:
                    self._log(f"⚠️  Could not create BM25 retriever: {e}")
                    self._log("Falling back to vector-only retrieval")
            else:
                self._log("⚠️  No nodes available for BM25, using vector-only retrieval")

            # 3. Create retriever (hybrid if BM25 available, otherwise vector-only)
            if bm25_retriever:
                # Use hybrid retriever with query fusion
                retriever = QueryFusionRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    similarity_top_k=5,
                    num_queries=1,  # No query generation, just fusion
                    mode="reciprocal_rerank",
                    use_async=True,
                    verbose=self.verbose,
                )
                self._log("Created hybrid retriever (vector + BM25)")
            else:
                # Fall back to vector-only retrieval
                retriever = vector_retriever
                self._log("Using vector-only retriever")

            # 4. Create chat summary memory buffer with automatic summarization
            memory = ChatSummaryMemoryBuffer.from_defaults(
                token_limit=3000,
                llm=self.llm,  # Uses LLM to summarize old messages
            )
            self._log(
                "Created chat summary memory buffer (token_limit=3000, auto-summarization enabled)"
            )

            # 5. Create chat engine with conversation history
            self.chat_engine = ContextChatEngine.from_defaults(
                retriever=retriever,
                memory=memory,
                llm=self.llm,
                system_prompt=self.system_prompt,
            )
            self._log("✅ Chat engine ready with conversation memory and summarization")

        except Exception as e:
            logger.error(f"Failed to create chat engine: {e}")
            raise

    def update_index(self, new_documents: list[Document]) -> None:
        """
        Add new documents to existing index.

        Documents are automatically persisted by Chroma's PersistentClient.

        Args:
            new_documents: List of new documents to add

        Example:
            >>> agent = ChatAgent(collection_name="stripe_api")
            >>> agent.update_index(new_docs)
        """
        if not self.index:
            raise ValueError("Index not initialized. Create index first.")

        try:
            self._log(f"Adding {len(new_documents)} new documents to index...")

            # Add documents to index (auto-persisted by Chroma)
            for doc in new_documents:
                self.index.insert(doc)

            self._log(f"✅ Added {len(new_documents)} documents")

            # Recreate chat engine to include new documents in BM25
            self._create_chat_engine()

        except Exception as e:
            logger.error(f"Failed to update index: {e}")
            raise

    def delete_collection(self) -> None:
        """
        Delete the Chroma collection (for testing/reset purposes).

        Warning: This permanently deletes all indexed documents!

        Example:
            >>> agent = ChatAgent(collection_name="test_api")
            >>> agent.delete_collection()  # Deletes collection from disk
        """
        if not self.chroma_client:
            raise ValueError("Chroma client not initialized")

        try:
            self._log(f"Deleting collection: {self.collection_name}")
            self.chroma_client.delete_collection(name=self.collection_name)
            self._log(f"✅ Deleted collection: {self.collection_name}")

            # Clear references
            self.index = None
            self.chat_engine = None
            self.chroma_collection = None

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    async def execute(
        self,
        user_query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a query against the indexed API documentation.

        This is the main interface implementing BaseAgent.execute().
        Uses conversational chat engine with memory to maintain context
        across multiple queries.

        Steps:
        1. Retrieve relevant documents using hybrid search (BM25 + vector)
        2. Generate response using LLM with retrieved context and conversation history
        3. Extract cURL commands from response (if any)
        4. Format source documents with metadata

        Args:
            user_query: Natural language query from user
            conversation_history: Previous conversation context (managed by chat engine)

        Returns:
            Dictionary with:
            - response: LLM-generated answer
            - curl_command: Extracted cURL command (or None)
            - sources: List of source documents with scores

        Example:
            >>> agent = ChatAgent(collection_name="stripe_api")
            >>> result = await agent.execute("How do I create a customer?")
            >>> print(result["response"])
            >>> print(result["curl_command"])
            >>>
            >>> # Follow-up question uses conversation history automatically
            >>> result2 = await agent.execute("What about updating one?")
        """
        if not self.chat_engine:
            raise ValueError("Chat engine not initialized. Build or load index first.")

        try:
            self._log(f"Processing query: {user_query}")

            # Step 1: Retrieve relevant documents and generate response with conversation history
            response = await self.chat_engine.achat(user_query)
            self._log(f"Generated response ({len(response.response)} chars)")

            # Step 2: Extract cURL command from response
            curl_command = self._extract_curl_command(response.response)
            if curl_command:
                self._log(f"Extracted cURL command: {curl_command[:50]}...")

            # Step 3: Format sources
            sources = self._format_sources(response.source_nodes)
            self._log(f"Retrieved {len(sources)} source documents")

            return {
                "response": response.response,
                "curl_command": curl_command,
                "sources": sources,
                "metadata": {
                    "query": user_query,
                    "num_sources": len(sources),
                    "has_curl": curl_command is not None,
                },
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    async def query(self, user_query: str) -> dict[str, Any]:
        """
        Alias for execute() for convenience.

        Args:
            user_query: Natural language query from user

        Returns:
            Dictionary with response, curl_command, and sources
        """
        return await self.execute(user_query)

    def _extract_curl_command(self, response_text: str) -> str | None:
        """
        Extract cURL command from LLM response using regex.

        Looks for patterns like:
        - curl https://...
        - curl -X POST https://...
        - ```bash\\ncurl ...\\n```

        Args:
            response_text: LLM response text

        Returns:
            Extracted cURL command or None if not found

        Example:
            >>> text = "Here's the command:\\n```bash\\ncurl -X GET https://api.stripe.com/v1/customers\\n```"
            >>> cmd = agent._extract_curl_command(text)
            >>> print(cmd)  # "curl -X GET https://api.stripe.com/v1/customers"
        """
        # Pattern 1: Code block with curl command
        code_block_pattern = r"```(?:bash|sh|shell)?\s*\n?(curl\s+[^\n`]+)"
        match = re.search(code_block_pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Pattern 2: Inline curl command
        inline_pattern = r"(curl\s+(?:-[a-zA-Z]\s+\S+\s+)*https?://[^\s]+(?:\s+-[a-zA-Z]\s+\S+)*)"
        match = re.search(inline_pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern 3: Multi-line curl command with backslashes
        multiline_pattern = r"(curl\s+[^`\n]*(?:\\\s*\n\s*[^`\n]+)*)"
        match = re.search(multiline_pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            # Clean up backslashes and newlines
            cmd = match.group(1).strip()
            cmd = re.sub(r"\\\s*\n\s*", " ", cmd)
            return cmd.strip()

        return None

    def _format_sources(self, source_nodes: list[NodeWithScore]) -> list[dict[str, Any]]:
        """
        Format source nodes with text, score, and metadata.

        Args:
            source_nodes: List of retrieved nodes with scores

        Returns:
            List of formatted source dictionaries

        Example:
            >>> sources = agent._format_sources(response.source_nodes)
            >>> for src in sources:
            ...     print(f"{src['score']:.3f}: {src['text'][:50]}...")
        """
        formatted_sources = []

        for i, node in enumerate(source_nodes):
            source = {
                "rank": i + 1,
                "score": node.score if node.score else 0.0,
                "text": node.node.get_content()[:500],  # First 500 chars
                "metadata": node.node.metadata,
            }

            # Add URL if available
            if "url" in node.node.metadata:
                source["url"] = node.node.metadata["url"]

            # Add page type if available
            if "page_type" in node.node.metadata:
                source["page_type"] = node.node.metadata["page_type"]

            formatted_sources.append(source)

        return formatted_sources

    def reset_conversation(self) -> None:
        """
        Reset conversation history.

        Clears the chat summary memory buffer (including both recent messages
        and any summarized history) to start a fresh conversation.

        Example:
            >>> agent = ChatAgent(collection_name="stripe_api")
            >>> await agent.execute("How do I create a customer?")
            >>> await agent.execute("What about updating one?")  # Uses history
            >>> agent.reset_conversation()  # Clear all history and summaries
            >>> await agent.execute("How do I delete a customer?")  # Fresh start
        """
        if self.chat_engine and hasattr(self.chat_engine, "reset"):
            self.chat_engine.reset()
            self._log("✅ Conversation history and summaries reset")
        else:
            self._log("⚠️ Chat engine does not support reset")
