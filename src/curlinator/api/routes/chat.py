"""
Chat endpoint for querying indexed documentation.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func

from curlinator.api.models.chat import (
    ChatRequest,
    ChatResponse,
    SessionResponse,
    SessionDetailResponse,
    SessionMessage
)
from curlinator.api.database import get_db
from curlinator.api.db.models import (
    DocumentationCollection,
    User,
    ChatSession,
    ChatMessage as DBChatMessage,
    SharePermission,
)
from curlinator.api.auth import get_current_user
from curlinator.api.utils.embeddings import get_embedding_model
from curlinator.api.middleware import limiter
from curlinator.agents.chat_agent import ChatAgent
from curlinator.api.routes.collections import get_accessible_collection
from curlinator.api.metrics import (
    chat_queries_total,
    chat_query_duration_seconds,
    chat_messages_total,
    vectorstore_queries_total
)
import time

router = APIRouter(prefix="/api/v1", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("60/minute")
async def chat(
    http_request: Request,
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Query indexed documentation using natural language with persistent conversation history.

    This endpoint requires JWT authentication.

    This endpoint:
    1. Verifies user authentication
    2. Queries the database to get the collection's embedding model metadata
    3. Creates or loads a ChatSession for conversation history
    4. Loads conversation history from the database
    5. Loads the same embedding model that was used during crawling
    6. Loads the ChatAgent with the specified collection and embedding model
    7. Processes the user query with conversation history
    8. Saves user message and assistant response to the database
    9. Returns response, cURL command, sources, and session_id

    Args:
        request: ChatRequest with collection_name, message, and optional session_id
        db: Database session dependency
        current_user: Authenticated user from JWT token

    Returns:
        ChatResponse with response, curl_command, sources, and session_id

    Raises:
        HTTPException: If query fails, collection not found, or user not authenticated
    """
    # Start timer for query duration
    start_time = time.time()

    try:
        # Get correlation ID from request state
        correlation_id = getattr(http_request.state, 'correlation_id', 'N/A')
        log_adapter = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})

        log_adapter.info(f"Chat query for collection: {request.collection_name}")

        # Check if user has access to the collection (owner, shared with CHAT permission, or public)
        collection, user_permission = get_accessible_collection(
            db,
            request.collection_name,
            current_user,
            required_permission=SharePermission.CHAT  # Require CHAT permission
        )

        log_adapter.info(f"Found collection in database: {collection.name}")
        log_adapter.info(f"Embedding provider: {collection.embedding_provider}")
        log_adapter.info(f"Embedding model: {collection.embedding_model}")

        # Get or create chat session
        chat_session = None
        if request.session_id:
            # Load existing session
            chat_session = db.query(ChatSession).filter(
                ChatSession.id == request.session_id,
                ChatSession.user_id == current_user.id,
                ChatSession.collection_id == collection.id
            ).first()

            if not chat_session:
                log_adapter.warning(f"Session not found: {request.session_id}")
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "Session not found",
                        "message": f"Chat session '{request.session_id}' not found or you don't have access to it",
                        "suggestion": "Start a new conversation by omitting the session_id"
                    }
                )
            log_adapter.info(f"Loaded existing session: {chat_session.id}")
        else:
            # Create new session
            chat_session = ChatSession(
                user_id=current_user.id,
                collection_id=collection.id
            )
            db.add(chat_session)
            db.commit()
            db.refresh(chat_session)
            log_adapter.info(f"Created new session: {chat_session.id}")

        # Load conversation history from database
        db_messages = db.query(DBChatMessage).filter(
            DBChatMessage.session_id == chat_session.id
        ).order_by(DBChatMessage.created_at).all()

        # Convert to ChatAgent format
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in db_messages
        ]

        # If conversation_history is provided in request (deprecated), use it instead
        # This maintains backward compatibility
        if request.conversation_history:
            log_adapter.warning("Using deprecated conversation_history field. Please use session_id instead.")
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]

        log_adapter.info(f"Loaded {len(history)} messages from conversation history")

        # Load the same embedding model that was used during crawling
        try:
            embed_model, provider_name, model_name = get_embedding_model(collection.embedding_provider)
            log_adapter.info(f"Loaded embedding model: {provider_name} / {model_name}")
        except ValueError as e:
            log_adapter.error(f"Failed to load embedding model: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Embedding model initialization failed",
                    "message": f"Failed to load embedding model: {str(e)}",
                    "suggestion": "Check that required API keys are set if the collection uses OPENAI or GEMINI embeddings"
                }
            )
        except Exception as e:
            log_adapter.error(f"Unexpected error loading embedding model: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Embedding model initialization failed",
                    "message": f"An unexpected error occurred while loading the embedding model: {str(e)}",
                    "suggestion": "Please try again. If the problem persists, contact support."
                }
            )

        # Load ChatAgent with existing collection and correct embedding model
        try:
            chat_agent = ChatAgent(
                collection_name=request.collection_name,
                embed_model=embed_model,
                verbose=True,
            )
            log_adapter.info(f"Successfully loaded ChatAgent for collection: {request.collection_name}")
        except Exception as e:
            log_adapter.error(f"Failed to load collection from vector store: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Collection not found in vector store",
                    "message": f"Collection '{request.collection_name}' exists in database but not in vector store",
                    "suggestion": "The collection may be corrupted. Try re-crawling the documentation."
                }
            )

        # Execute query (async method)
        try:
            # Track vector store query
            vectorstore_queries_total.labels(collection=request.collection_name).inc()

            result = await chat_agent.execute(
                user_query=request.message,
                conversation_history=history,
            )

            # Track successful query
            query_duration = time.time() - start_time
            chat_query_duration_seconds.observe(query_duration)
            chat_queries_total.labels(collection=request.collection_name, status="success").inc()

        except Exception as e:
            log_adapter.error(f"Query execution failed: {str(e)}", exc_info=True)
            chat_queries_total.labels(collection=request.collection_name, status="failure").inc()
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Query execution failed",
                    "message": f"Failed to process your query: {str(e)}",
                    "suggestion": "Try rephrasing your question or check if the collection contains relevant information"
                }
            )

        # Save user message to database
        user_message = DBChatMessage(
            session_id=chat_session.id,
            role="user",
            content=request.message
        )
        db.add(user_message)
        chat_messages_total.labels(role="user").inc()

        # Save assistant response to database
        assistant_message = DBChatMessage(
            session_id=chat_session.id,
            role="assistant",
            content=result.get("response", ""),
            curl_command=result.get("curl_command")
        )
        db.add(assistant_message)
        chat_messages_total.labels(role="assistant").inc()

        # Commit messages to database
        db.commit()
        log_adapter.info(f"Saved 2 messages to session {chat_session.id}")

        return ChatResponse(
            response=result.get("response", ""),
            curl_command=result.get("curl_command"),
            sources=result.get("sources", []),
            session_id=chat_session.id,
        )

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        log_adapter.error(f"Database error during chat query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Database error",
                "message": "Failed to retrieve collection information from database",
                "suggestion": "This may be a temporary issue. Please try again."
            }
        )
    except Exception as e:
        db.rollback()
        log_adapter.error(f"Unexpected error during chat query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": f"An unexpected error occurred: {str(e)}",
                "suggestion": "Please try again. If the problem persists, contact support."
            }
        )



@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List all chat sessions for the authenticated user.

    Returns a list of sessions with basic information (no messages).
    Use GET /api/v1/sessions/{session_id} to get full session details with messages.

    Args:
        db: Database session dependency
        current_user: Authenticated user from JWT token

    Returns:
        List of SessionResponse objects
    """
    try:
        logger.info(f"Listing sessions for user: {current_user.email}")

        # Query all sessions for the user with message counts in a single query
        # This avoids N+1 query problem by using a LEFT JOIN and GROUP BY
        sessions_with_counts = db.query(
            ChatSession,
            func.count(DBChatMessage.id).label('message_count')
        ).outerjoin(
            DBChatMessage,
            ChatSession.id == DBChatMessage.session_id
        ).filter(
            ChatSession.user_id == current_user.id
        ).group_by(
            ChatSession.id
        ).order_by(
            ChatSession.updated_at.desc()
        ).all()

        # Build response with message counts
        result = []
        for session, message_count in sessions_with_counts:
            result.append(SessionResponse(
                id=session.id,
                collection_name=session.collection.name,
                collection_id=session.collection_id,
                message_count=message_count,
                created_at=session.created_at,
                updated_at=session.updated_at
            ))

        logger.info(f"Found {len(result)} sessions for user {current_user.email}")
        return result

    except SQLAlchemyError as e:
        logger.error(f"Database error listing sessions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Database error",
                "message": "Failed to retrieve sessions from database",
                "suggestion": "This may be a temporary issue. Please try again."
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error listing sessions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": f"An unexpected error occurred: {str(e)}",
                "suggestion": "Please try again. If the problem persists, contact support."
            }
        )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get detailed information about a specific chat session including all messages.

    Args:
        session_id: Session ID
        db: Database session dependency
        current_user: Authenticated user from JWT token

    Returns:
        SessionDetailResponse with full message history

    Raises:
        HTTPException: If session not found or user doesn't have access
    """
    try:
        logger.info(f"Getting session: {session_id}")

        # Query session and verify ownership
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        ).first()

        if not session:
            logger.warning(f"Session not found: {session_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Session not found",
                    "message": f"Chat session '{session_id}' not found or you don't have access to it",
                    "suggestion": "Use GET /api/v1/sessions to see your available sessions"
                }
            )

        # Load messages
        messages = db.query(DBChatMessage).filter(
            DBChatMessage.session_id == session_id
        ).order_by(DBChatMessage.created_at).all()

        # Convert to response format
        message_list = [
            SessionMessage(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                curl_command=msg.curl_command,
                created_at=msg.created_at
            )
            for msg in messages
        ]

        return SessionDetailResponse(
            id=session.id,
            collection_name=session.collection.name,
            collection_id=session.collection_id,
            messages=message_list,
            created_at=session.created_at,
            updated_at=session.updated_at
        )

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error getting session: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Database error",
                "message": "Failed to retrieve session from database",
                "suggestion": "This may be a temporary issue. Please try again."
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error getting session: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": f"An unexpected error occurred: {str(e)}",
                "suggestion": "Please try again. If the problem persists, contact support."
            }
        )



@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a chat session and all its messages.

    Args:
        session_id: Session ID
        db: Database session dependency
        current_user: Authenticated user from JWT token

    Returns:
        Success message

    Raises:
        HTTPException: If session not found or user doesn't have access
    """
    try:
        logger.info(f"Deleting session: {session_id}")

        # Query session and verify ownership
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        ).first()

        if not session:
            logger.warning(f"Session not found: {session_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Session not found",
                    "message": f"Chat session '{session_id}' not found or you don't have access to it",
                    "suggestion": "Use GET /api/v1/sessions to see your available sessions"
                }
            )

        # Delete session (cascade will delete messages)
        db.delete(session)
        db.commit()

        logger.info(f"Deleted session: {session_id}")
        return {"message": "Session deleted successfully", "session_id": session_id}

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error deleting session: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Database error",
                "message": "Failed to delete session from database",
                "suggestion": "This may be a temporary issue. Please try again."
            }
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error deleting session: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": f"An unexpected error occurred: {str(e)}",
                "suggestion": "Please try again. If the problem persists, contact support."
            }
        )


@router.post("/sessions/{session_id}/reset")
async def reset_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Reset a chat session by deleting all its messages.

    The session itself is preserved, but all conversation history is cleared.

    Args:
        session_id: Session ID
        db: Database session dependency
        current_user: Authenticated user from JWT token

    Returns:
        Success message

    Raises:
        HTTPException: If session not found or user doesn't have access
    """
    try:
        logger.info(f"Resetting session: {session_id}")

        # Query session and verify ownership
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        ).first()

        if not session:
            logger.warning(f"Session not found: {session_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Session not found",
                    "message": f"Chat session '{session_id}' not found or you don't have access to it",
                    "suggestion": "Use GET /api/v1/sessions to see your available sessions"
                }
            )

        # Delete all messages in the session
        message_count = db.query(DBChatMessage).filter(
            DBChatMessage.session_id == session_id
        ).delete()

        db.commit()

        logger.info(f"Reset session {session_id}: deleted {message_count} messages")
        return {
            "message": "Session reset successfully",
            "session_id": session_id,
            "messages_deleted": message_count
        }

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error resetting session: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Database error",
                "message": "Failed to reset session in database",
                "suggestion": "This may be a temporary issue. Please try again."
            }
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error resetting session: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": f"An unexpected error occurred: {str(e)}",
                "suggestion": "Please try again. If the problem persists, contact support."
            }
        )


