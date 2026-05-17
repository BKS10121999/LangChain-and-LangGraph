import operator
import re
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from prompts import MEMORY_SUMMARY_PROMPT, get_chat_prompt_template, get_intent_classification_prompt
from schemas import AnswerResponse, CalculationResponse, SummarizationResponse, UpdateMemoryResponse, UserIntent


class AgentState(TypedDict, total=False):
    """
    JSON-safe graph state.

    Pydantic objects are converted to dictionaries before they enter state so
    LangGraph checkpoints do not need to deserialize project-local classes such
    as `schemas.UserIntent`.
    """

    user_input: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]

    intent: Dict[str, Any]
    intent_type: str
    next_step: str

    conversation_history: List[Dict[str, Any]]
    conversation_summary: str
    memory_documents: List[str]
    active_documents: List[str]

    retrieval_plan: Dict[str, Any]
    retrieval_results: List[Dict[str, Any]]
    retrieval_diagnostics: Dict[str, Any]
    current_sources: List[str]

    current_response: Optional[Dict[str, Any]]
    tools_used: List[str]

    session_id: Optional[str]
    user_id: Optional[str]

    actions_taken: Annotated[List[str], operator.add]


def invoke_react_agent(response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools) -> Tuple[Dict[str, Any], List[str]]:
    """Invoke a ReAct agent and collect concrete tool calls for observability."""
    agent = create_react_agent(model=llm, tools=tools, response_format=response_schema)
    result = agent.invoke({"messages": messages})
    tools_used = [
        getattr(message, "name", None) or getattr(message, "tool_call_id", "tool")
        for message in result.get("messages", [])
        if isinstance(message, ToolMessage)
    ]
    return result, tools_used


def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """Classify the request and reset all request-scoped state."""
    llm = _config(config, "llm")
    user_input = state.get("user_input") or ""

    # Pure arithmetic should not enter the RAG path. This gate runs before
    # retriever query planning so expressions like "25 * 4" never produce
    # document search diagnostics or accidental source citations.
    if is_pure_math_query(user_input):
        return {
            "actions_taken": ["classify_intent"],
            "intent": {
                "intent_type": "calculation",
                "confidence": 0.98,
                "reasoning": "Arithmetic-only query; route directly to calculator without retrieval.",
            },
            "intent_type": "calculation",
            "next_step": "pure_math_agent",
            "retrieval_plan": {"pure_math": True, "requires_documents": False},
            "retrieval_results": [],
            "retrieval_diagnostics": {"bypassed": True, "reason": "pure_math_query"},
            "current_sources": [],
            "current_response": None,
            "tools_used": [],
        }

    history_text = _format_history(state)

    prompt = get_intent_classification_prompt().format(
        user_input=user_input,
        conversation_history=history_text,
    )

    try:
        raw = llm.with_structured_output(UserIntent).invoke(prompt)
        intent_obj = raw if isinstance(raw, UserIntent) else UserIntent(**_as_dict(raw))
    except Exception as exc:
        intent_obj = _heuristic_intent(user_input, f"LLM classifier fallback: {exc}")

    intent = intent_obj.model_dump(mode="json")
    intent_type = intent.get("intent_type") or "unknown"

    retriever = _config(config, "retriever")
    retrieval_plan = retriever.analyze_query(user_input, intent_type) if retriever else {}

    if retrieval_plan.get("ambiguous"):
        next_step = "clarification_agent"
    elif intent_type == "summarization":
        next_step = "summarization_agent"
    elif intent_type == "calculation":
        next_step = "calculation_agent"
    elif intent_type == "unknown":
        next_step = "out_of_scope_agent"
    else:
        next_step = "qa_agent"

    return {
        "actions_taken": ["classify_intent"],
        "intent": intent,
        "intent_type": intent_type,
        "next_step": next_step,
        "retrieval_plan": retrieval_plan,
        "retrieval_results": [],
        "retrieval_diagnostics": {},
        "current_sources": [],
        "current_response": None,
        "tools_used": [],
    }


def clarification_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """Ask for missing document context instead of guessing."""
    question = "Which document, invoice, contract, or claim are you referring to?"
    if "amount" in (state.get("user_input") or "").lower():
        question = "Which document or invoice amount are you referring to?"

    response = AnswerResponse(
        question=state.get("user_input") or "",
        answer=question,
        sources=[],
        confidence=0.95,
    ).model_dump(mode="json")

    return {
        "messages": [AIMessage(content=question)],
        "actions_taken": ["clarification_agent"],
        "current_response": response,
        "current_sources": [],
        "active_documents": [],
        "next_step": "update_memory",
    }


def out_of_scope_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """Handle non-document requests without pretending RAG found nothing."""
    answer = (
        "I am set up to answer questions, summarize documents, and perform calculations "
        "over the document collection. Please ask about an invoice, claim, contract, "
        "report, or a calculation."
    )
    response = AnswerResponse(
        question=state.get("user_input") or "",
        answer=answer,
        sources=[],
        confidence=0.9,
    ).model_dump(mode="json")
    return _agent_return(
        "out_of_scope_agent",
        response,
        [AIMessage(content=answer)],
        [],
        [],
        [],
        {"bypassed": True, "reason": "unknown_or_out_of_scope_intent"},
    )


def pure_math_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """Evaluate arithmetic-only input with the calculator and no retrieval."""
    query = state.get("user_input") or ""
    expression = normalize_pure_math_expression(query)
    calculator = _find_tool(_config(config, "tools") or [], "calculator")

    if calculator:
        result = calculator.invoke({"expression": expression})
        tools_used = ["calculator"]
    else:
        result = "Calculator tool is not configured."
        tools_used = []

    explanation = f"Evaluated `{expression}` with the calculator. Result: {result}."
    response = CalculationResponse(
        expression=expression,
        result=str(result),
        explanation=explanation,
        confidence=0.98,
    ).model_dump(mode="json")

    return _agent_return(
        "pure_math_agent",
        response,
        [AIMessage(content=explanation)],
        tools_used,
        [],
        [],
        {"bypassed": True, "reason": "pure_math_query"},
    )


def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """Answer factual questions using request-scoped retrieved context."""
    llm = _config(config, "llm")
    tools = _config(config, "tools") or []
    chunks, diagnostics = _retrieve_for_state(state, config, "qa")
    sources = _doc_ids(chunks)

    if not chunks:
        return _not_found_response(state, "I could not find a matching document for that question.", diagnostics)

    if _query_needs_explicit_field(state.get("user_input") or "", "diagnosis", chunks):
        answer = "The patient's diagnosis was not found in the retrieved medical or claim documents."
        response = AnswerResponse(
            question=state.get("user_input") or "",
            answer=answer,
            sources=[],
            confidence=0.25,
        ).model_dump(mode="json")
        diagnostics["searched_sources"] = sources
        return _agent_return("qa_agent", response, [AIMessage(content=answer)], [], [], chunks, diagnostics)

    deterministic = _deterministic_lookup(state.get("user_input") or "", chunks)
    if deterministic:
        response = AnswerResponse(
            question=state.get("user_input") or "",
            answer=deterministic,
            sources=sources,
            confidence=0.9,
        ).model_dump(mode="json")
        return _agent_return("qa_agent", response, [AIMessage(content=deterministic)], [], sources, chunks, diagnostics)

    messages = _messages_with_context("qa", state, chunks)
    result, tools_used = invoke_react_agent(AnswerResponse, messages, llm, tools)
    response = _serialize_response(result.get("structured_response"))
    response_sources = _clean_sources(response.get("sources"), sources)
    if response_sources:
        response["sources"] = response_sources
    return _agent_return("qa_agent", response, result.get("messages", []), tools_used, response_sources, chunks, diagnostics)


def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """Summarize only matching documents and fail closed when none exist."""
    llm = _config(config, "llm")
    tools = _config(config, "tools") or []
    chunks, diagnostics = _retrieve_for_state(state, config, "summarization")
    sources = _doc_ids(chunks)

    if not chunks:
        answer = "No matching document was found to summarize. Please provide a document ID or a more specific document type."
        response = SummarizationResponse(
            original_length=0,
            summary=answer,
            key_points=[],
            document_ids=[],
            confidence=0.95,
        ).model_dump(mode="json")
        return _agent_return("summarization_agent", response, [AIMessage(content=answer)], [], [], chunks, diagnostics)

    messages = _messages_with_context("summarization", state, chunks)
    result, tools_used = invoke_react_agent(SummarizationResponse, messages, llm, tools)
    response = _serialize_response(result.get("structured_response"))
    response_sources = _clean_sources(response.get("document_ids"), sources)
    if response_sources:
        response["document_ids"] = response_sources
    return _agent_return("summarization_agent", response, result.get("messages", []), tools_used, response_sources, chunks, diagnostics)


def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Run a deterministic retrieval -> extraction -> calculator chain.

    The LLM is not asked to discover totals from scratch; code extracts one
    reliable amount per retrieved document and the calculator tool performs the
    arithmetic. This makes totals reproducible and keeps sources request-scoped.
    """
    retriever = _config(config, "retriever")
    tools = _config(config, "tools") or []
    chunks, diagnostics = _retrieve_for_state(state, config, "calculation")
    sources = _doc_ids(chunks)

    if not chunks:
        return _not_found_response(state, "I could not find matching documents to calculate from.", diagnostics, action="calculation_agent")

    values = retriever.extract_financial_values(chunks) if retriever else []
    if not values:
        answer = "I found matching documents, but I could not identify reliable numeric values to calculate."
        response = CalculationResponse(expression="", result="", explanation=answer).model_dump(mode="json")
        diagnostics["extracted_values"] = []
        return _agent_return("calculation_agent", response, [AIMessage(content=answer)], [], sources, chunks, diagnostics)

    expression = " + ".join(_format_number(value["amount"]) for value in values)
    calculator = _find_tool(tools, "calculator")
    if calculator:
        result = calculator.invoke({"expression": expression})
        tools_used = ["calculator"]
    else:
        result = str(sum(value["amount"] for value in values))
        tools_used = []

    value_lines = [
        f"{value['doc_id']} {value['label']}: {_format_number(value['amount'])}"
        for value in values
    ]
    explanation = (
        "I retrieved the matching documents, extracted one reliable amount from each, "
        f"and evaluated `{expression}` with the calculator. Result: {result}.\n"
        + "\n".join(value_lines)
    )
    response = CalculationResponse(
        expression=expression,
        result=str(result),
        explanation=explanation,
        confidence=0.9 if values else 0.0,
    ).model_dump(mode="json")
    diagnostics["extracted_values"] = values

    return _agent_return("calculation_agent", response, [AIMessage(content=explanation)], tools_used, sources, chunks, diagnostics)


def update_memory(state: AgentState, config: RunnableConfig) -> AgentState:
    """Update long-lived summary without polluting current-turn sources."""
    llm = _config(config, "llm")
    current_sources = state.get("current_sources") or []
    memory_documents = _unique([*(state.get("memory_documents") or []), *current_sources])
    history_text = _format_messages(state.get("messages") or [])

    prompt = (
        f"{MEMORY_SUMMARY_PROMPT}\n\n"
        f"Previous summary:\n{state.get('conversation_summary', '')}\n\n"
        f"Recent Conversation:\n{history_text}\n\n"
        f"Current-turn sources only: {current_sources}\n"
        f"Known memory documents: {memory_documents}"
    )

    try:
        raw = llm.with_structured_output(UpdateMemoryResponse).invoke(prompt)
        memory = raw if isinstance(raw, UpdateMemoryResponse) else UpdateMemoryResponse(**_as_dict(raw))
        summary = memory.summary
        memory_documents = _unique([*memory_documents, *memory.document_ids])
    except Exception:
        summary = state.get("conversation_summary") or "No previous conversation."

    return {
        "conversation_summary": summary,
        "memory_documents": memory_documents,
        "actions_taken": ["update_memory"],
        "next_step": "end",
    }


def should_continue(state: AgentState) -> str:
    return state.get("next_step", "end")


def create_workflow(llm, tools):
    """Create and compile the LangGraph workflow with checkpoint memory."""
    workflow = StateGraph(AgentState)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("clarification_agent", clarification_agent)
    workflow.add_node("out_of_scope_agent", out_of_scope_agent)
    workflow.add_node("pure_math_agent", pure_math_agent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "clarification_agent": "clarification_agent",
            "out_of_scope_agent": "out_of_scope_agent",
            "pure_math_agent": "pure_math_agent",
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent",
            "end": END,
        },
    )
    workflow.add_edge("clarification_agent", "update_memory")
    workflow.add_edge("out_of_scope_agent", "update_memory")
    workflow.add_edge("pure_math_agent", "update_memory")
    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")
    workflow.add_edge("update_memory", END)
    return workflow.compile(checkpointer=InMemorySaver())


def _retrieve_for_state(state: AgentState, config: RunnableConfig, intent_type: str) -> Tuple[List[Any], Dict[str, Any]]:
    retriever = _config(config, "retriever")
    if not retriever:
        return [], {"error": "Retriever is not configured."}
    plan = state.get("retrieval_plan") or retriever.analyze_query(state.get("user_input") or "", intent_type)
    chunks, diagnostics = retriever.retrieve(
        state.get("user_input") or "",
        intent_type=intent_type,
        top_k=plan.get("top_k"),
        filters=plan,
    )
    return chunks, diagnostics


def _messages_with_context(intent_type: str, state: AgentState, chunks: List[Any]) -> List[BaseMessage]:
    context = _format_chunks(chunks)
    prompt_template = get_chat_prompt_template(intent_type)
    messages = prompt_template.invoke({
        "input": state.get("user_input", ""),
        "chat_history": state.get("messages", []),
    }).to_messages()
    # Context is injected as an explicit system message so the LLM answers only
    # from current-turn retrieval rather than older checkpoint sources.
    messages.insert(0, SystemMessage(content=f"Current request retrieved context:\n{context}"))
    return messages


def _agent_return(
    action: str,
    response: Dict[str, Any],
    messages: List[BaseMessage],
    tools_used: List[str],
    sources: List[str],
    chunks: List[Any],
    diagnostics: Dict[str, Any],
) -> AgentState:
    return {
        "messages": messages,
        "actions_taken": [action],
        "current_response": response,
        "tools_used": tools_used,
        "current_sources": sources,
        "active_documents": sources,
        "retrieval_results": _chunk_records(chunks),
        "retrieval_diagnostics": diagnostics,
        "next_step": "update_memory",
    }


def _not_found_response(state: AgentState, answer: str, diagnostics: Dict[str, Any], action: str = "qa_agent") -> AgentState:
    response = AnswerResponse(
        question=state.get("user_input") or "",
        answer=answer,
        sources=[],
        confidence=0.15,
    ).model_dump(mode="json")
    return _agent_return(action, response, [AIMessage(content=answer)], [], [], [], diagnostics)


def _heuristic_intent(user_input: str, reason: str = "") -> UserIntent:
    text = user_input.lower()
    document_terms = [
        "invoice", "invoices", "claim", "claims", "contract", "contracts",
        "agreement", "report", "document", "financial", "finance",
        "quarterly", "annual",
    ]
    calculation_terms = ["calculate", "sum", "total", "average", "difference", "add", "revenue", "value"]
    if any(term in text for term in ["summarize", "summarise", "summary"]):
        intent_type = "summarization"
        confidence = 0.8
    elif any(term in text for term in calculation_terms) and (
        any(term in text for term in document_terms) or any(term in text for term in ["calculate", "sum", "average", "difference", "add"])
    ):
        intent_type = "calculation"
        confidence = 0.78
    elif not _looks_like_document_request(text):
        intent_type = "unknown"
        confidence = 0.75
    else:
        intent_type = "qa"
        confidence = 0.65
    return UserIntent(intent_type=intent_type, confidence=confidence, reasoning=reason or "Heuristic intent classification.")


def _looks_like_document_request(text: str) -> bool:
    document_terms = {
        "invoice", "invoices", "claim", "claims", "contract", "contracts",
        "agreement", "report", "reports", "document", "documents", "patient",
        "diagnosis", "medical", "healthcare", "client", "customer", "claimant",
        "policy", "revenue", "amount", "total", "status", "date",
    }
    return bool(
        re.search(r"\b(?:inv|con|clm)-\d+\b", text)
        or any(re.search(rf"\b{re.escape(term)}\b", text) for term in document_terms)
    )


def is_pure_math_query(query: str) -> bool:
    """
    Return True only for arithmetic that can be answered without documents.

    The detector intentionally rejects document/entity words before looking at
    operators and numeric density. This keeps "total invoice revenue" in the
    RAG calculation path while allowing "15% of 250000" to bypass retrieval.
    """
    text = _strip_math_request_words((query or "").strip().lower())
    if not text:
        return False

    document_terms = {
        "invoice", "invoices", "claim", "claims", "contract", "contracts",
        "agreement", "report", "reports", "document", "documents", "client",
        "customer", "patient", "claimant", "policy", "revenue", "value",
        "balance", "medical",
    }
    if re.search(r"\b(?:inv|con|clm)-\d+\b", text):
        return False
    if any(re.search(rf"\b{re.escape(term)}\b", text) for term in document_terms):
        return False

    math_keywords = {"sqrt", "square root", "percent", "percentage", "of"}
    allowed_words = {"sqrt", "of", "percent", "percentage", "plus", "minus", "times", "divided", "by"}
    words = re.findall(r"[a-zA-Z]+", text)
    if any(word not in allowed_words for word in words):
        return False

    normalized = normalize_pure_math_expression(text)
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    has_operator = bool(re.search(r"[\+\-\*/%^()]", text))
    has_math_keyword = any(keyword in text for keyword in math_keywords)
    numeric_chars = len(re.findall(r"[\d\.\+\-\*/%^()]", text))
    numeric_density = numeric_chars / max(len(text.replace(" ", "")), 1)

    safe_expression = re.fullmatch(r"[\d\s\.\+\-\*/%\^\(\)]+|sqrt\s*\([\d\s\.]+\)", normalized) is not None
    return bool(numbers and safe_expression and (has_operator or has_math_keyword or numeric_density >= 0.65))


def normalize_pure_math_expression(query: str) -> str:
    """Convert user-friendly arithmetic into calculator syntax."""
    expression = _strip_math_request_words((query or "").strip().lower())
    expression = expression.replace("^", "**")
    expression = re.sub(r"\bplus\b", "+", expression)
    expression = re.sub(r"\bminus\b", "-", expression)
    expression = re.sub(r"\btimes\b", "*", expression)
    expression = re.sub(r"\bdivided\s+by\b", "/", expression)
    expression = re.sub(
        r"(\d+(?:\.\d+)?)\s*(?:%|percent|percentage)\s+of\s+(\d+(?:\.\d+)?)",
        r"(\1 / 100) * \2",
        expression,
    )
    expression = expression.replace(" ", "")
    return expression


def _strip_math_request_words(query: str) -> str:
    """Remove harmless request wording before pure-math detection."""
    text = query.strip().rstrip("?!.")
    text = re.sub(r"^(please\s+)?(calculate|compute|evaluate|solve|what\s+is|what's|find)\s+", "", text)
    text = re.sub(r"^(the\s+)?", "", text)
    return text.strip()


def _query_needs_explicit_field(query: str, field: str, chunks: List[Any]) -> bool:
    if field not in query.lower():
        return False
    return not any(field in (chunk.content or "").lower() for chunk in chunks)


def _deterministic_lookup(query: str, chunks: List[Any]) -> Optional[str]:
    """Answer high-confidence field lookups directly from metadata."""
    if len(_doc_ids(chunks)) != 1:
        return None
    chunk = chunks[0]
    metadata = chunk.metadata or {}
    query_lower = query.lower()

    field_map = [
        (["client", "customer"], ["customer_name", "client"]),
        (["claimant", "patient"], ["patient_name", "claimant", "customer_name"]),
        (["status"], ["status"]),
        (["date"], ["date"]),
    ]
    for query_terms, metadata_fields in field_map:
        if any(term in query_lower for term in query_terms):
            for field in metadata_fields:
                value = metadata.get(field)
                if value:
                    return f"The {query_terms[0]} in {chunk.doc_id} is {value}."

    if any(term in query_lower for term in ["amount", "total", "value"]):
        for field in ["total", "amount", "value"]:
            if field in metadata:
                return f"The {field} for {chunk.doc_id} is ${float(metadata[field]):,.2f}."
    return None


def _format_chunks(chunks: List[Any]) -> str:
    parts = []
    for chunk in chunks:
        metadata = chunk.metadata or {}
        parts.append(
            "\n".join([
                f"Document ID: {chunk.doc_id}",
                f"Type: {metadata.get('doc_type')}",
                f"Category: {metadata.get('category')}",
                f"Title: {metadata.get('title')}",
                f"Customer/Patient: {metadata.get('customer_name') or metadata.get('patient_name') or metadata.get('claimant')}",
                f"Relevance: {chunk.relevance_score:.3f}",
                f"Content: {(chunk.content or '').strip()}",
            ])
        )
    return "\n\n---\n\n".join(parts)


def _chunk_records(chunks: List[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "doc_id": chunk.doc_id,
            "doc_type": (chunk.metadata or {}).get("doc_type"),
            "category": (chunk.metadata or {}).get("category"),
            "score": round(float(chunk.relevance_score), 4),
            "title": (chunk.metadata or {}).get("title"),
        }
        for chunk in chunks
    ]


def _doc_ids(chunks: List[Any]) -> List[str]:
    return _unique([chunk.doc_id for chunk in chunks])


def _clean_sources(candidate_sources: Any, allowed_sources: List[str]) -> List[str]:
    if not candidate_sources:
        return allowed_sources
    allowed = set(allowed_sources)
    return [source for source in candidate_sources if source in allowed]


def _format_history(state: AgentState) -> str:
    saved = state.get("conversation_history") or []
    saved_lines = [f"{item.get('type', 'Message')}: {item.get('content', '')}" for item in saved[-8:]]
    return "\n".join([*saved_lines, _format_messages(state.get("messages") or [])]).strip()


def _format_messages(messages: List[BaseMessage]) -> str:
    lines = []
    for message in messages:
        role = message.__class__.__name__
        content = getattr(message, "content", None) or getattr(message, "text", None) or str(message)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _serialize_response(response: Any) -> Dict[str, Any]:
    if response is None:
        return {}
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if isinstance(response, dict):
        return response
    return {"response": str(response)}


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _config(config: RunnableConfig, key: str) -> Any:
    return (config.get("configurable") or {}).get(key)


def _find_tool(tools: List[Any], name: str) -> Optional[Any]:
    for tool in tools:
        if getattr(tool, "name", None) == name:
            return tool
    return None


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _unique(values: List[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))
