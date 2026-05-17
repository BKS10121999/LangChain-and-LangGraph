from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal, Annotated
from datetime import datetime, timezone


class DocumentChunk(BaseModel):
    """Represents a chunk of document content"""
    doc_id: str = Field(description="Document identifier")
    content: str = Field(description="The actual text content")
    metadata: Dict[str, Any] = Field(default_factory=lambda: dict, description="Additional metadata")
    relevance_score: float = Field(default=0.0, description="Relevance score for retrieval")


class AnswerResponse(BaseModel):
    """Structured response for Q&A tasks.

    Fields:
    - question: original user question
    - answer: assistant's answer text
    - sources: list of source document ids or citations
    - confidence: float in [0, 1]
    - timestamp: UTC time when the answer was generated
    """
    question: str = Field(default="", description="The user's original question")
    answer: str = Field(default="", description="The generated answer text")
    sources: List[str] = Field(default_factory=list, description="List of source document IDs or citations")
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="Confidence score between 0 and 1",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp when the answer was generated")



class SummarizationResponse(BaseModel):
    """Structured response for summarization tasks"""
    original_length: int = Field(description="Length of original text")
    summary: str = Field(description="The generated summary")
    key_points: List[str] = Field(description="List of key points extracted")
    document_ids: List[str] = Field(default_factory=lambda: list, description="Documents summarized")
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="Confidence score based on retrieval quality and answer completeness",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CalculationResponse(BaseModel):
    """Structured response for calculation tasks"""
    expression: str = Field(description="The mathematical expression")
    result: str = Field(description="The calculated result as returned by the calculator tool")
    explanation: str = Field(description="Step-by-step explanation")
    units: Optional[str] = Field(default=None, description="Units if applicable")
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="Confidence score based on numeric extraction quality",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UpdateMemoryResponse(BaseModel):
    """Response after updating memory"""
    summary: str = Field(description="Summary of the conversation up to this point")
    document_ids: List[str] = Field(default_factory=lambda: list, description="List of documents ids that are relevant to the users last message")


class UserIntent(BaseModel):
    """User intent classification schema.

    Fields:
    - intent_type: one of 'qa', 'summarization', 'calculation', or 'unknown'
    - confidence: float in [0, 1]
    - reasoning: explanation for the classification
    """
    intent_type: Literal["qa", "summarization", "calculation", "unknown"] = Field(
        default="unknown",
        description="Predicted intent type",
    )
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="Confidence for the predicted intent",
    )
    reasoning: str = Field(default="", description="Textual reasoning or evidence for the classification")


class SessionState(BaseModel):
    """Session state"""
    session_id: str
    user_id: str
    conversation_history: List[Dict[str, Any]] = Field(default_factory=lambda: list)
    document_context: List[str] = Field(default_factory=lambda: list, description="Active document IDs")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


if __name__ == "__main__":
    # Example usage / quick smoke tests
    example_answer = AnswerResponse(
        question="What is LangGraph?",
        answer="LangGraph is a hypothetical assistant for document QA and retrieval.",
        sources=["doc_123", "doc_456"],
        confidence=0.92,
    )

    example_intent = UserIntent(
        intent_type="qa",
        confidence=0.95,
        reasoning="User asked a direct question about the doc contents.",
    )

    def _print(obj):
        if hasattr(obj, "model_dump_json"):
            print(obj.model_dump_json(indent=2))
        else:
            print(obj.json(indent=2))

    print("--- AnswerResponse example ---")
    _print(example_answer)
    print("--- UserIntent example ---")
    _print(example_intent)
