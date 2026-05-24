"""
Evaluation and Report Generation for the EcoHome AI Energy Optimization Agent.

Provides:
- Response quality scoring (accuracy, relevance, completeness, usefulness)
- Tool usage evaluation (appropriateness, completeness)
- Comprehensive report generation with aggregation, analysis, and formatting
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv()

_eval_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY") or os.getenv("OPENAI_API_KEY"),
)

AVAILABLE_TOOLS = [
    "get_weather_forecast",
    "get_electricity_prices",
    "query_energy_usage",
    "query_historical_energy_usage",
    "query_solar_generation",
    "analyze_solar_generation",
    "get_recent_energy_summary",
    "search_energy_tips",
    "calculate_energy_savings",
]


def evaluate_response(
    question: str,
    actual_response: str,
    expected_response: str,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    keyword_overlap = _compute_keyword_overlap(actual_response, expected_response)
    semantic_eval = _semantic_evaluate(
        question=question,
        actual_response=actual_response,
        expected_response=expected_response,
        context=context,
    )
    metrics = semantic_eval.get("metrics", {})
    overall_score = _compute_overall_score(metrics)

    return {
        "question": question,
        "metrics": metrics,
        "overall_score": overall_score,
        "keyword_overlap": keyword_overlap,
        "feedback": semantic_eval.get("feedback", ""),
        "strengths": semantic_eval.get("strengths", []),
        "weaknesses": semantic_eval.get("weaknesses", []),
    }


def _semantic_evaluate(
    question: str,
    actual_response: str,
    expected_response: str,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    eval_prompt = f"""You are an expert evaluator for an AI energy optimization assistant.

Score the ACTUAL RESPONSE against the EXPECTED RESPONSE on these metrics (1-10 each):

1. Accuracy - Are facts, numbers, and claims correct? Does it avoid hallucinations?
2. Relevance - Does it directly address the user's question without going off-topic?
3. Completeness - Does it cover all key points from the expected response?
4. Usefulness - Would a homeowner find this actionable and helpful?

USER QUESTION:
{question}

EXPECTED RESPONSE:
{expected_response}

ACTUAL RESPONSE:
{actual_response}

{f'ADDITIONAL CONTEXT: {context}' if context else ''}

Respond in this EXACT format (no other text):
ACCURACY: <score>
RELEVANCE: <score>
COMPLETENESS: <score>
USEFULNESS: <score>
FEEDBACK: <2-3 sentence overall assessment>
STRENGTHS: <comma-separated list>
WEAKNESSES: <comma-separated list>"""

    try:
        response = _eval_llm.invoke([HumanMessage(content=eval_prompt)])
        return _parse_evaluation_response(response.content)
    except Exception as e:
        return {
            "metrics": {
                "accuracy": _fallback_score(actual_response, expected_response),
                "relevance": _fallback_score(actual_response, expected_response),
                "completeness": _fallback_score(actual_response, expected_response),
                "usefulness": _fallback_score(actual_response, expected_response),
            },
            "feedback": f"Semantic evaluation failed ({type(e).__name__}). Using keyword-based fallback.",
            "strengths": [],
            "weaknesses": ["Could not perform semantic evaluation"],
        }


def _parse_evaluation_response(text: str) -> Dict[str, Any]:
    def extract_score(label: str) -> int:
        match = re.search(rf"{label}:\s*(\d+)", text, re.IGNORECASE)
        if match:
            return max(1, min(10, int(match.group(1))))
        return 5

    def extract_text(label: str) -> str:
        match = re.search(rf"{label}:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def extract_list(label: str) -> List[str]:
        raw = extract_text(label)
        if not raw or raw.lower() in ("none", "n/a", ""):
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]

    return {
        "metrics": {
            "accuracy": extract_score("ACCURACY"),
            "relevance": extract_score("RELEVANCE"),
            "completeness": extract_score("COMPLETENESS"),
            "usefulness": extract_score("USEFULNESS"),
        },
        "feedback": extract_text("FEEDBACK"),
        "strengths": extract_list("STRENGTHS"),
        "weaknesses": extract_list("WEAKNESSES"),
    }


def _compute_keyword_overlap(actual: str, expected: str) -> float:
    actual_tokens = set(re.findall(r"\b\w+\b", actual.lower()))
    expected_tokens = set(re.findall(r"\b\w+\b", expected.lower()))
    if not expected_tokens:
        return 1.0 if not actual_tokens else 0.0
    overlap = actual_tokens & expected_tokens
    return round(len(overlap) / len(expected_tokens), 4)


def _compute_overall_score(metrics: Dict[str, int]) -> float:
    weights = {
        "accuracy": 0.30,
        "relevance": 0.25,
        "completeness": 0.25,
        "usefulness": 0.20,
    }
    total = sum(metrics.get(key, 5) * weight for key, weight in weights.items())
    return round(total, 2)


def _fallback_score(actual: str, expected: str) -> int:
    overlap = _compute_keyword_overlap(actual, expected)
    return max(1, min(10, int(overlap * 10)))


def evaluate_tool_usage(
    question: str,
    messages: List[Any],
    expected_tools: List[str],
    tool_logs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    actual_tools = _extract_tools_from_messages(messages)

    if tool_logs:
        for log in tool_logs:
            tool_name = log.get("tool", "")
            if tool_name and tool_name not in actual_tools:
                actual_tools.append(tool_name)

    expected_set = set(expected_tools)
    actual_set = set(actual_tools)
    correct_tools = expected_set & actual_set
    missing_tools = expected_set - actual_set
    extra_tools = actual_set - expected_set
    invalid_tools = actual_set - set(AVAILABLE_TOOLS)

    completeness_ratio = len(correct_tools) / len(expected_set) if expected_set else (1.0 if not actual_set else 0.5)
    completeness_score = max(1, min(10, round(completeness_ratio * 10)))
    appropriateness_score = _compute_appropriateness(actual_tools, expected_tools, extra_tools, invalid_tools)
    feedback = _generate_tool_feedback(correct_tools, missing_tools, extra_tools, invalid_tools, tool_logs)

    return {
        "question": question,
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "metrics": {
            "tool_appropriateness": appropriateness_score,
            "tool_completeness": completeness_score,
        },
        "details": {
            "correct": sorted(correct_tools),
            "missing": sorted(missing_tools),
            "extra": sorted(extra_tools),
            "invalid": sorted(invalid_tools),
        },
        "feedback": feedback,
    }


def _extract_tools_from_messages(messages: List[Any]) -> List[str]:
    tools_called: List[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "")
                if tool_name:
                    tools_called.append(tool_name)
        elif isinstance(msg, ToolMessage):
            pass
    return tools_called


def _compute_appropriateness(
    actual_tools: List[str],
    expected_tools: List[str],
    extra_tools: set,
    invalid_tools: set,
) -> int:
    if not actual_tools:
        return 3 if expected_tools else 10

    score = 10 - len(invalid_tools) * 3 - len(extra_tools)
    if not extra_tools and not invalid_tools:
        score = 10
    return max(1, min(10, score))


def _generate_tool_feedback(
    correct_tools: set,
    missing_tools: set,
    extra_tools: set,
    invalid_tools: set,
    tool_logs: Optional[List[Dict[str, Any]]] = None,
) -> str:
    lines: List[str] = []
    if not missing_tools and not extra_tools and not invalid_tools:
        lines.append("Excellent: All expected tools were called with no unnecessary extras.")
    else:
        if correct_tools:
            lines.append(f"Correctly used: {', '.join(sorted(correct_tools))}")
        if missing_tools:
            lines.append(f"Missing (should have been called): {', '.join(sorted(missing_tools))}")
        if extra_tools:
            lines.append(f"Extra (not strictly needed): {', '.join(sorted(extra_tools))}")
        if invalid_tools:
            lines.append(f"Invalid (not a real tool): {', '.join(sorted(invalid_tools))}")

    if tool_logs:
        errors = [log for log in tool_logs if log.get("is_error")]
        if errors:
            lines.append("Tool execution errors: " + ", ".join(log["tool"] for log in errors))

    return " ".join(lines) if lines else "No tool usage feedback available."


SCORE_BANDS = {
    (9, 10): ("Excellent", "High confidence and strong coverage."),
    (7, 8): ("Good", "Solid quality with minor gaps."),
    (5, 6): ("Adequate", "Partially meets expectations but needs improvement."),
    (3, 4): ("Poor", "Significant gaps reduce reliability."),
    (1, 2): ("Critical", "Major issues need attention before relying on results."),
}


def interpret_score(score: float) -> Dict[str, str]:
    """Interpret a numeric score using rubric-style bands."""
    rounded = round(score)
    for (low, high), (label, description) in SCORE_BANDS.items():
        if low <= rounded <= high:
            return {"label": label, "description": description}
    return {"label": "Unknown", "description": "Score is outside the expected range."}


class EvaluationReportGenerator:
    """Aggregate evaluation outputs into markdown and console-friendly reports."""

    def __init__(self, project_name: str = "EcoHome AI Energy Optimization Agent"):
        self.project_name = project_name
        self.response_evaluations: List[Dict[str, Any]] = []
        self.tool_evaluations: List[Dict[str, Any]] = []

    def add_response_eval(self, evaluation: Dict[str, Any]) -> None:
        """Store a single response-quality evaluation."""
        self.response_evaluations.append(evaluation)

    def add_tool_eval(self, evaluation: Dict[str, Any]) -> None:
        """Store a single tool-usage evaluation."""
        self.tool_evaluations.append(evaluation)

    def generate_report(self) -> Dict[str, Any]:
        """Build the aggregated report data structure."""
        response_metrics = self._aggregate_response_metrics()
        tool_metrics = self._aggregate_tool_metrics()
        overall_score = self._compute_overall_score(response_metrics, tool_metrics)

        strengths = self._collect_common_items(self.response_evaluations, "strengths")
        weaknesses = self._collect_common_items(self.response_evaluations, "weaknesses")
        recommendations = self._generate_recommendations(
            response_metrics=response_metrics,
            tool_metrics=tool_metrics,
            weaknesses=weaknesses,
        )

        return {
            "project_name": self.project_name,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "response_count": len(self.response_evaluations),
                "tool_count": len(self.tool_evaluations),
                "overall_score": overall_score,
                "interpretation": interpret_score(overall_score),
            },
            "response_metrics": response_metrics,
            "tool_metrics": tool_metrics,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
        }

    def format_console(self) -> str:
        """Return a plain-text report suitable for notebook or terminal output."""
        report = self.generate_report()
        summary = report["summary"]

        lines = [
            "=" * 72,
            f"Evaluation Report: {report['project_name']}",
            "=" * 72,
            f"Generated At: {report['generated_at']}",
            f"Overall Score: {summary['overall_score']}/10",
            f"Interpretation: {summary['interpretation']['label']} - {summary['interpretation']['description']}",
            "",
            "Response Metrics:",
            self._format_metric_table(report["response_metrics"]),
            "",
            "Tool Metrics:",
            self._format_metric_table(report["tool_metrics"]),
            "",
            "Strengths:",
        ]

        if report["strengths"]:
            lines.extend(f"- {item}" for item in report["strengths"])
        else:
            lines.append("- None identified")

        lines.append("")
        lines.append("Weaknesses:")
        if report["weaknesses"]:
            lines.extend(f"- {item}" for item in report["weaknesses"])
        else:
            lines.append("- None identified")

        lines.append("")
        lines.append("Recommendations:")
        if report["recommendations"]:
            lines.extend(f"- {item}" for item in report["recommendations"])
        else:
            lines.append("- No recommendations. Performance is stable across measured metrics.")

        lines.append("=" * 72)
        return "\n".join(lines)

    def format_markdown(self) -> str:
        """Return a markdown-formatted report."""
        report = self.generate_report()
        summary = report["summary"]

        parts = [
            f"# Evaluation Report: {report['project_name']}",
            "",
            f"Generated At: {report['generated_at']}",
            "",
            "## Summary",
            f"- Overall Score: {summary['overall_score']}/10",
            f"- Interpretation: {summary['interpretation']['label']} - {summary['interpretation']['description']}",
            f"- Response Evaluations: {summary['response_count']}",
            f"- Tool Evaluations: {summary['tool_count']}",
            "",
            "## Response Metrics",
            self._format_metric_markdown(report["response_metrics"]),
            "",
            "## Tool Metrics",
            self._format_metric_markdown(report["tool_metrics"]),
            "",
            "## Strengths",
        ]

        if report["strengths"]:
            parts.extend(f"- {item}" for item in report["strengths"])
        else:
            parts.append("- None identified")

        parts.append("")
        parts.append("## Weaknesses")
        if report["weaknesses"]:
            parts.extend(f"- {item}" for item in report["weaknesses"])
        else:
            parts.append("- None identified")

        parts.append("")
        parts.append("## Recommendations")
        if report["recommendations"]:
            parts.extend(f"- {item}" for item in report["recommendations"])
        else:
            parts.append("- No recommendations. Performance is stable across measured metrics.")

        return "\n".join(parts)

    def _aggregate_response_metrics(self) -> Dict[str, Any]:
        """Average response metrics across collected evaluations."""
        metric_names = ["accuracy", "relevance", "completeness", "usefulness"]
        return self._aggregate_metric_group(self.response_evaluations, metric_names)

    def _aggregate_tool_metrics(self) -> Dict[str, Any]:
        """Average tool metrics across collected evaluations."""
        metric_names = ["tool_appropriateness", "tool_completeness"]
        return self._aggregate_metric_group(self.tool_evaluations, metric_names)

    def _aggregate_metric_group(
        self,
        evaluations: List[Dict[str, Any]],
        metric_names: List[str],
    ) -> Dict[str, Any]:
        """Aggregate a homogeneous metric group into averages and interpretations."""
        if not evaluations:
            return {
                "count": 0,
                "metrics": {name: 0.0 for name in metric_names},
                "average": 0.0,
                "interpretation": interpret_score(0.0),
            }

        totals = {name: 0.0 for name in metric_names}
        for evaluation in evaluations:
            for name in metric_names:
                totals[name] += float(evaluation.get("metrics", {}).get(name, 0))

        averages = {
            name: round(totals[name] / len(evaluations), 2)
            for name in metric_names
        }
        group_average = round(sum(averages.values()) / len(metric_names), 2)

        return {
            "count": len(evaluations),
            "metrics": averages,
            "average": group_average,
            "interpretation": interpret_score(group_average),
        }

    def _compute_overall_score(
        self,
        response_metrics: Dict[str, Any],
        tool_metrics: Dict[str, Any],
    ) -> float:
        """Compute overall score using response-first weighting."""
        response_average = float(response_metrics.get("average", 0.0))
        tool_average = float(tool_metrics.get("average", 0.0))

        if response_metrics.get("count", 0) and tool_metrics.get("count", 0):
            return round((response_average * 0.7) + (tool_average * 0.3), 2)
        if response_metrics.get("count", 0):
            return response_average
        if tool_metrics.get("count", 0):
            return tool_average
        return 0.0

    def _collect_common_items(
        self,
        evaluations: List[Dict[str, Any]],
        field_name: str,
    ) -> List[str]:
        """Collect and deduplicate list items from evaluation fields."""
        items: List[str] = []
        for evaluation in evaluations:
            value = evaluation.get(field_name, [])
            if isinstance(value, list):
                items.extend(item for item in value if item)
            elif isinstance(value, str) and value:
                items.append(value)
        return sorted(dict.fromkeys(items))

    def _generate_recommendations(
        self,
        response_metrics: Dict[str, Any],
        tool_metrics: Dict[str, Any],
        weaknesses: List[str],
    ) -> List[str]:
        """Generate actionable recommendations from aggregated results."""
        recommendations: List[str] = []

        if response_metrics.get("metrics", {}).get("accuracy", 10) < 7:
            recommendations.append(
                "Improve factual grounding in final responses and explicitly tie claims to tool outputs."
            )
        if response_metrics.get("metrics", {}).get("completeness", 10) < 7:
            recommendations.append(
                "Expand final recommendations to cover all expected points, including schedules, savings, and environmental impact."
            )
        if tool_metrics.get("metrics", {}).get("tool_completeness", 10) < 7:
            recommendations.append(
                "Review planner routing rules so all necessary tools are called before reasoning starts."
            )
        if tool_metrics.get("metrics", {}).get("tool_appropriateness", 10) < 7:
            recommendations.append(
                "Reduce unnecessary tool calls and keep routing aligned to the question type."
            )

        if not recommendations and weaknesses:
            recommendations.append(
                "Address the recurring weaknesses identified in the qualitative feedback."
            )
        if not recommendations:
            recommendations.append(
                "No major issues detected. Keep monitoring output quality with the current rubric."
            )

        return recommendations

    def _format_metric_table(self, metric_group: Dict[str, Any]) -> str:
        """Format metric data for console output."""
        metrics = metric_group.get("metrics", {})
        if not metrics:
            return "No metrics collected."

        rows = [f"  {name}: {score}/10" for name, score in metrics.items()]
        rows.append(f"  Average: {metric_group.get('average', 0.0)}/10")
        rows.append(
            f"  Interpretation: {metric_group.get('interpretation', {}).get('label', 'Unknown')}"
        )
        return "\n".join(rows)

    def _format_metric_markdown(self, metric_group: Dict[str, Any]) -> str:
        """Format metric data for markdown output."""
        metrics = metric_group.get("metrics", {})
        if not metrics:
            return "No metrics collected."

        rows = ["| Metric | Score |", "| --- | ---: |"]
        rows.extend(f"| {name} | {score}/10 |" for name, score in metrics.items())
        rows.append(f"| Average | {metric_group.get('average', 0.0)}/10 |")
        return "\n".join(rows)
