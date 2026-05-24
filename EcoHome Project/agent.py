"""
EcoHome Smart Energy Optimization Agent

Production-style LangGraph StateGraph agent with:
- Explicit tool routing with category-based selection
- Multi-step reasoning with chain-of-thought grounding
- Anti-hallucination guardrails (data-only assertions)
- Structured outputs with typed state
- Comprehensive error handling with retry + circuit breaker
- Detailed structured logging with trace IDs
"""

import os
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from tools import TOOL_KIT

load_dotenv()

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger("ecohome_agent")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-8s | %(name)s | trace=%(trace_id)s | %(message)s",
            defaults={"trace_id": "none"},
        )
    )
    logger.addHandler(_handler)


class _TraceAdapter(logging.LoggerAdapter):
    """Adapter that injects trace_id into every log record."""

    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})
        kwargs["extra"]["trace_id"] = self.extra.get("trace_id", "none")
        return msg, kwargs


# ---------------------------------------------------------------------------
# Tool Routing Categories
# ---------------------------------------------------------------------------
TOOL_CATEGORIES: Dict[str, List[str]] = {
    "weather": ["get_weather_forecast"],
    "pricing": ["get_electricity_prices"],
    "usage_history": ["query_energy_usage", "query_historical_energy_usage", "get_recent_energy_summary"],
    "solar": ["query_solar_generation", "analyze_solar_generation"],
    "knowledge": ["search_energy_tips"],
    "calculation": ["calculate_energy_savings"],
}

TOOL_DESCRIPTIONS_FOR_ROUTING: str = """Available tool categories and when to use them:
- weather: Get forecast data (solar irradiance, temperature for HVAC load). Use when question involves future planning, solar predictions, or HVAC scheduling.
- pricing: Get electricity price schedules (time-of-use rates, peak/off-peak). Use when question involves cost optimization, appliance scheduling, or billing.
- usage_history: Query past energy consumption from the database. Use when question involves trends, comparisons, device-level analysis, or anomaly detection.
- solar: Query solar panel generation and battery data. Use when question involves solar output, self-consumption, or grid export.
- knowledge: Search RAG vector store for energy-saving tips and best practices. Use for general advice, device recommendations, or optimization strategies.
- calculation: Compute savings estimates. Use after gathering baseline data to quantify recommendations.

ROUTING RULES:
1. For optimization questions: ALWAYS gather usage_history first, then pricing, then optionally weather.
2. For forecasting questions: ALWAYS use weather + solar together.
3. For cost questions: ALWAYS use pricing + usage_history together.
4. For general tips: Use knowledge first; only add other tools if specific data is needed.
5. Call calculate_energy_savings ONLY after you have real numbers from other tools.
6. NEVER fabricate data. If a tool returns an error, acknowledge the gap."""


# ---------------------------------------------------------------------------
# Agent State Schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """Typed state schema carried through the graph."""

    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    context: Optional[str]
    plan: Optional[str]
    tools_called: List[str]
    tool_results: List[Dict[str, Any]]
    tool_errors: List[Dict[str, str]]
    reasoning: Optional[str]
    final_response: Optional[str]
    error: Optional[str]
    iteration: int
    trace_id: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TOOL_ITERATIONS = 4
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5  # seconds


# ---------------------------------------------------------------------------
# Prompts (separated for maintainability)
# ---------------------------------------------------------------------------
PLANNER_SYSTEM_TEMPLATE = """{instructions}

{tool_routing_guide}

PLANNING INSTRUCTIONS:
- You are the planning module. Your job is to decide WHICH tools to call and with WHAT arguments.
- Think step-by-step about what data you need to answer the user's question.
- Call multiple tools in a single turn when they are independent (e.g., weather + pricing).
- Do NOT generate a final answer here. Only plan and call tools, or indicate you have enough data.
- If you already have sufficient data from previous tool calls, respond with a brief summary of what you know (no tool calls) so the reasoner can take over.

{iteration_context}
{additional_context}"""

REASONER_SYSTEM_PROMPT = """You are the analytical reasoning module for an EcoHome energy optimization assistant.

STRICT RULES:
1. Base ALL claims on data from the tool results in this conversation. NEVER invent numbers.
2. If data is missing or a tool failed, explicitly state what is unknown.
3. When estimating savings, show your calculation steps (e.g., "X kWh × $Y/kWh = $Z").
4. For CO2 estimates, use 0.42 kg CO2/kWh (US grid average) unless location-specific data is available.
5. For scheduling, reference actual price periods from the pricing data.

STRUCTURE your analysis as:
## Data Summary
- List key metrics from tool results (with sources)

## Analysis
- Patterns, anomalies, or insights derived from the data

## Recommendations
- Numbered list of actionable recommendations
- Each with: action, expected savings (kWh and $), confidence level (high/medium/low)

## Environmental Impact
- CO2 reduction estimate with calculation shown

## Optimal Schedule
- Time-based recommendations tied to pricing data

## Limitations
- What data was unavailable or uncertain"""

FINAL_RESPONSE_SYSTEM_PROMPT = """You are the EcoHome Smart Energy Advisor delivering your final response to a homeowner.

RULES:
1. Rewrite the reasoning analysis into a clear, friendly, professional response.
2. NEVER show raw JSON, internal tool names, or system details.
3. All numbers must come from the reasoning analysis. Do NOT add numbers that aren't there.
4. Use clear section headers, bullet points, and formatting for readability.
5. Include a brief "What I Analyzed" section so users understand the data basis.
6. End with a prioritized "Next Steps" section.
7. If anything was uncertain or unavailable, be transparent about it.
8. Keep the tone helpful and encouraging — energy optimization should feel achievable.

FORMAT:
- Use markdown-style headers (##) for sections
- Use bullet points for lists
- Bold key numbers and savings figures
- Keep total response under 800 words unless the question requires more detail"""


# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------
class Agent:
    """EcoHome Energy Optimization Agent powered by LangGraph StateGraph."""

    def __init__(self, instructions: str, model: str = "gpt-4o-mini"):
        self.instructions = instructions
        self.model_name = model

        self.llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            base_url="https://openai.vocareum.com/v1",
            api_key=os.getenv("VOCAREUM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        )

        # Bind tools for tool-calling
        self.tools = TOOL_KIT
        self.tools_by_name: Dict[str, Any] = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the compiled graph
        self.graph = self._build_graph()

        logger.info(
            "Agent initialized | model=%s | tools=%d | categories=%s",
            model,
            len(self.tools),
            list(TOOL_CATEGORIES.keys()),
        )

    # ------------------------------------------------------------------
    # Graph Construction
    # ------------------------------------------------------------------
    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph StateGraph with nodes and conditional edges."""

        graph = StateGraph(AgentState)

        # Nodes
        graph.add_node("planner", self._planner_node)
        graph.add_node("tool_executor", self._tool_executor_node)
        graph.add_node("reasoner", self._reasoner_node)
        graph.add_node("final_response", self._final_response_node)

        # Entry
        graph.set_entry_point("planner")

        # Planner -> tool_executor OR reasoner
        graph.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {"tool_executor": "tool_executor", "reasoner": "reasoner"},
        )

        # Tool executor -> planner (loop) OR reasoner (done gathering)
        graph.add_conditional_edges(
            "tool_executor",
            self._route_after_tools,
            {"planner": "planner", "reasoner": "reasoner"},
        )

        # Reasoner -> final_response
        graph.add_edge("reasoner", "final_response")

        # Final response -> END
        graph.add_edge("final_response", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Routing Logic
    # ------------------------------------------------------------------
    def _route_after_planner(self, state: AgentState) -> Literal["tool_executor", "reasoner"]:
        """Route based on whether the planner issued tool calls."""
        last_message = state["messages"][-1]
        has_tool_calls = (
            isinstance(last_message, AIMessage)
            and hasattr(last_message, "tool_calls")
            and last_message.tool_calls
        )

        if has_tool_calls:
            tool_names = [tc["name"] for tc in last_message.tool_calls]
            self._log(state, "DEBUG", f"Planner -> tool_executor | tools={tool_names}")
            return "tool_executor"

        self._log(state, "DEBUG", "Planner -> reasoner (no tool calls)")
        return "reasoner"

    def _route_after_tools(self, state: AgentState) -> Literal["planner", "reasoner"]:
        """Decide if we need more tool calls or can proceed to reasoning."""

        # Hard cap on iterations
        if state["iteration"] >= MAX_TOOL_ITERATIONS:
            self._log(
                state, "WARNING",
                f"Max iterations reached ({MAX_TOOL_ITERATIONS}), forcing -> reasoner"
            )
            return "reasoner"

        # If all recent tool calls errored, don't loop — go reason with what we have
        recent_results = state["tool_results"][-len(state.get("tools_called", [])):]
        all_errors = all(
            isinstance(r.get("result"), dict) and "error" in r.get("result", {})
            for r in recent_results
        ) if recent_results else False

        if all_errors:
            self._log(state, "WARNING", "All recent tools errored -> reasoner (graceful)")
            return "reasoner"

        self._log(state, "DEBUG", f"Tool results collected (iter={state['iteration']}) -> planner")
        return "planner"

    # ------------------------------------------------------------------
    # Node: Planner
    # ------------------------------------------------------------------
    def _planner_node(self, state: AgentState) -> Dict[str, Any]:
        """Analyze the question and decide which tools to invoke next."""
        self._log(state, "INFO", f"PLANNER NODE | iteration={state['iteration']}")

        # Build iteration-aware context
        iteration_context = ""
        if state["iteration"] > 0:
            called = state.get("tools_called", [])
            errors = state.get("tool_errors", [])
            iteration_context = (
                f"You have completed {state['iteration']} tool-calling round(s).\n"
                f"Tools already called: {called}\n"
            )
            if errors:
                iteration_context += f"Tools that failed: {[e['tool'] for e in errors]}\n"
            iteration_context += (
                "Only call additional tools if essential information is still missing. "
                "Do NOT re-call tools that already succeeded."
            )

        additional_context = ""
        if state.get("context"):
            additional_context = f"User-provided context:\n{state['context']}"

        system_prompt = PLANNER_SYSTEM_TEMPLATE.format(
            instructions=self.instructions,
            tool_routing_guide=TOOL_DESCRIPTIONS_FOR_ROUTING,
            iteration_context=iteration_context,
            additional_context=additional_context,
        )

        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

        response = self._invoke_llm_with_retry(messages, use_tools=True, state=state)

        if response.tool_calls:
            self._log(
                state, "INFO",
                f"Planner requesting tools: {[tc['name'] for tc in response.tool_calls]}"
            )
        else:
            self._log(state, "INFO", "Planner decided: sufficient data, no more tools")

        return {
            "messages": [response],
            "plan": response.content if response.content else state.get("plan"),
        }

    # ------------------------------------------------------------------
    # Node: Tool Executor
    # ------------------------------------------------------------------
    def _tool_executor_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute tool calls, collect results, track errors."""
        self._log(state, "INFO", "TOOL EXECUTOR NODE")

        last_ai_message = state["messages"][-1]
        tool_calls = (
            last_ai_message.tool_calls
            if isinstance(last_ai_message, AIMessage) and hasattr(last_ai_message, "tool_calls")
            else []
        )

        tool_messages: List[ToolMessage] = []
        new_tool_results: List[Dict[str, Any]] = list(state.get("tool_results") or [])
        new_tools_called: List[str] = list(state.get("tools_called") or [])
        new_tool_errors: List[Dict[str, str]] = list(state.get("tool_errors") or [])

        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]
            call_id = call["id"]

            start_time = time.time()
            self._log(state, "INFO", f"Executing: {tool_name}({json.dumps(tool_args, default=str)})")

            result = self._execute_tool_with_retry(tool_name, tool_args, state)
            elapsed_ms = int((time.time() - start_time) * 1000)

            # Track execution
            new_tools_called.append(tool_name)

            # Detect tool-level errors
            is_error = isinstance(result, dict) and "error" in result
            if is_error:
                new_tool_errors.append({"tool": tool_name, "error": result["error"]})
                self._log(state, "WARNING", f"Tool {tool_name} failed: {result['error']} ({elapsed_ms}ms)")
            else:
                result_size = len(json.dumps(result, default=str))
                self._log(state, "INFO", f"Tool {tool_name} OK | {result_size} chars | {elapsed_ms}ms")

            # Serialize result for message history
            serialized = json.dumps(result, default=str)
            tool_messages.append(ToolMessage(content=serialized, tool_call_id=call_id))
            new_tool_results.append({
                "tool": tool_name,
                "args": tool_args,
                "result": result,
                "is_error": is_error,
                "elapsed_ms": elapsed_ms,
            })

        return {
            "messages": tool_messages,
            "tool_results": new_tool_results,
            "tools_called": new_tools_called,
            "tool_errors": new_tool_errors,
            "iteration": state["iteration"] + 1,
        }

    # ------------------------------------------------------------------
    # Node: Reasoner
    # ------------------------------------------------------------------
    def _reasoner_node(self, state: AgentState) -> Dict[str, Any]:
        """Synthesize tool results into structured, grounded analysis."""
        self._log(state, "INFO", "REASONER NODE")

        # Build a data summary preamble so the LLM sees what's available
        data_preamble = self._build_data_summary(state)

        reasoner_prompt = (
            REASONER_SYSTEM_PROMPT
            + "\n\n--- DATA AVAILABLE ---\n"
            + data_preamble
        )

        messages = [SystemMessage(content=reasoner_prompt)] + list(state["messages"])

        response = self._invoke_llm_with_retry(messages, use_tools=False, state=state)

        self._log(state, "INFO", f"Reasoner produced {len(response.content)} chars")

        return {
            "messages": [response],
            "reasoning": response.content,
        }

    # ------------------------------------------------------------------
    # Node: Final Response
    # ------------------------------------------------------------------
    def _final_response_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate the polished user-facing response."""
        self._log(state, "INFO", "FINAL RESPONSE NODE")

        messages = [SystemMessage(content=FINAL_RESPONSE_SYSTEM_PROMPT)] + list(state["messages"])

        response = self._invoke_llm_with_retry(messages, use_tools=False, state=state)

        self._log(state, "INFO", f"Final response: {len(response.content)} chars")

        return {
            "messages": [response],
            "final_response": response.content,
        }

    # ------------------------------------------------------------------
    # Helper: Build Data Summary for Reasoner
    # ------------------------------------------------------------------
    def _build_data_summary(self, state: AgentState) -> str:
        """Create a compact summary of available tool results for grounding."""
        tool_results = state.get("tool_results") or []
        if not tool_results:
            return "No tool data was collected. Base your response only on general knowledge and be transparent about limitations."

        lines = []
        for tr in tool_results:
            status = "ERROR" if tr.get("is_error") else "OK"
            lines.append(f"- {tr['tool']}({json.dumps(tr['args'], default=str)}) -> {status}")

        errors = state.get("tool_errors") or []
        if errors:
            lines.append(f"\nFailed tools ({len(errors)}): {[e['tool'] for e in errors]}")
            lines.append("For failed tools, do NOT guess what the data would have been.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helper: LLM Invocation with Retry + Exponential Backoff
    # ------------------------------------------------------------------
    def _invoke_llm_with_retry(
        self, messages: List[BaseMessage], use_tools: bool = False, state: Optional[AgentState] = None
    ) -> AIMessage:
        """Invoke the LLM with exponential backoff retry."""
        llm = self.llm_with_tools if use_tools else self.llm
        trace_id = state.get("trace_id", "none") if state else "none"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                start = time.time()
                response = llm.invoke(messages)
                elapsed_ms = int((time.time() - start) * 1000)

                self._log(
                    state, "DEBUG",
                    f"LLM call OK | attempt={attempt} | {elapsed_ms}ms | "
                    f"tools={'yes' if use_tools else 'no'} | "
                    f"response_len={len(response.content or '')}"
                )
                return response

            except Exception as e:
                wait_time = RETRY_BACKOFF_BASE ** attempt
                self._log(
                    state, "WARNING",
                    f"LLM call failed (attempt {attempt}/{MAX_RETRIES}): "
                    f"{type(e).__name__}: {str(e)[:200]} | retry in {wait_time:.1f}s"
                )
                if attempt == MAX_RETRIES:
                    self._log(state, "ERROR", "LLM invocation exhausted all retries")
                    return AIMessage(
                        content=(
                            "I'm experiencing a temporary issue connecting to my AI backend. "
                            "Please try again in a moment. If this persists, check your API configuration."
                        )
                    )
                time.sleep(wait_time)

    # ------------------------------------------------------------------
    # Helper: Tool Execution with Retry + Validation
    # ------------------------------------------------------------------
    def _execute_tool_with_retry(
        self, tool_name: str, tool_args: Dict, state: Optional[AgentState] = None
    ) -> Any:
        """Execute a tool with retry, argument validation, and error containment."""

        # Validate tool exists
        tool = self.tools_by_name.get(tool_name)
        if tool is None:
            available = list(self.tools_by_name.keys())
            self._log(state, "ERROR", f"Unknown tool '{tool_name}' | available={available}")
            return {
                "error": f"Tool '{tool_name}' does not exist. Available tools: {available}",
                "error_type": "ToolNotFound",
            }

        # Execute with retry
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = tool.invoke(tool_args)

                # Validate result is not None
                if result is None:
                    return {"error": f"Tool '{tool_name}' returned None.", "error_type": "EmptyResult"}

                return result

            except TypeError as e:
                # Likely bad arguments — don't retry
                self._log(state, "ERROR", f"Tool '{tool_name}' bad args: {e}")
                return {
                    "error": f"Invalid arguments for '{tool_name}': {str(e)}",
                    "error_type": "InvalidArguments",
                }

            except Exception as e:
                last_error = e
                self._log(
                    state, "WARNING",
                    f"Tool '{tool_name}' attempt {attempt}/{MAX_RETRIES} failed: "
                    f"{type(e).__name__}: {str(e)[:150]}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_BASE * attempt)

        return {
            "error": f"Tool '{tool_name}' failed after {MAX_RETRIES} attempts: {str(last_error)}",
            "error_type": type(last_error).__name__,
        }

    # ------------------------------------------------------------------
    # Helper: Structured Logging
    # ------------------------------------------------------------------
    def _log(self, state: Optional[AgentState], level: str, message: str) -> None:
        """Emit a log message with trace_id from state."""
        trace_id = state.get("trace_id", "none") if state else "none"
        log_fn = getattr(logger, level.lower(), logger.debug)
        log_fn(message, extra={"trace_id": trace_id})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def invoke(
        self,
        question: str,
        context: str = None,
        return_state: bool = False,
    ) -> Any:
        """
        Run the agent on a user question about energy optimization.

        Args:
            question: The user's natural language question.
            context: Optional additional context (e.g., user profile, location).
            return_state: When True, return structured execution data including
                messages, tool logs, reasoning, and final response.

        Returns:
            The agent's final response string by default, or a structured dict
            when return_state is True.
        """
        trace_id = uuid.uuid4().hex[:12]

        logger.info(
            "Agent invoked | question='%s' | has_context=%s",
            question[:100],
            bool(context),
            extra={"trace_id": trace_id},
        )

        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "context": context,
            "plan": None,
            "tools_called": [],
            "tool_results": [],
            "tool_errors": [],
            "reasoning": None,
            "final_response": None,
            "error": None,
            "iteration": 0,
            "trace_id": trace_id,
        }

        try:
            start_time = time.time()
            final_state = self.graph.invoke(initial_state)
            total_ms = int((time.time() - start_time) * 1000)

            # Log execution summary
            logger.info(
                "Agent complete | iterations=%d | tools_called=%s | total_time=%dms",
                final_state.get("iteration", 0),
                final_state.get("tools_called", []),
                total_ms,
                extra={"trace_id": trace_id},
            )

            final_response = final_state.get("final_response")

            if not final_response:
                for msg in reversed(final_state["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_response = msg.content
                        break

            if not final_response:
                final_response = (
                    "I was unable to generate a complete response. "
                    "Please try rephrasing your question or providing more details."
                )

            if return_state:
                return {
                    "question": question,
                    "context": context,
                    "trace_id": trace_id,
                    "final_response": final_response,
                    "messages": final_state.get("messages", []),
                    "tools_called": final_state.get("tools_called", []),
                    "tool_results": final_state.get("tool_results", []),
                    "tool_errors": final_state.get("tool_errors", []),
                    "reasoning": final_state.get("reasoning"),
                    "iteration": final_state.get("iteration", 0),
                    "error": final_state.get("error"),
                }

            return final_response

        except Exception as e:
            logger.exception(
                "Agent execution failed: %s: %s",
                type(e).__name__,
                str(e),
                extra={"trace_id": trace_id},
            )
            error_response = (
                "I encountered an unexpected error while processing your request. "
                "Please try again. If the problem persists, it may be a configuration issue."
            )
            if return_state:
                return {
                    "question": question,
                    "context": context,
                    "trace_id": trace_id,
                    "final_response": error_response,
                    "messages": [],
                    "tools_called": [],
                    "tool_results": [],
                    "tool_errors": [
                        {"tool": "agent", "error": f"{type(e).__name__}: {str(e)}"}
                    ],
                    "reasoning": None,
                    "iteration": 0,
                    "error": f"{type(e).__name__}: {str(e)}",
                }
            return error_response

    def get_agent_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools_by_name.keys())

    def get_tool_categories(self) -> Dict[str, List[str]]:
        """Get tool routing categories for inspection."""
        return TOOL_CATEGORIES.copy()
