# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import os
import re
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.models import Gemini
from google.adk.workflow import START, Edge, Workflow, node
from google.genai import types

from budget_validation_agent.models import (
    PurchaseRequest,
    RiskAssessment,
    ValidationResult,
)

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
try:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
except Exception:
    config = {
        "model_name": "gemini-3.1-flash-lite",
        "budget_validation_rule": {"always_require_review": False},
    }


@node
def parse_input(ctx: Context, node_input: Any) -> PurchaseRequest:
    """Parses and validates the input request.
    Handles Pub/Sub JSON structures (with base64 data) or plain JSON.
    """
    text = ""
    # Extract text from node_input if it's types.Content or similar
    if isinstance(node_input, types.Content):
        if node_input.parts:
            text = node_input.parts[0].text or ""
    elif isinstance(node_input, dict):
        text = json.dumps(node_input)
    elif isinstance(node_input, str):
        text = node_input

    # Clean markdown code blocks if the input is wrapped in them
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data_dict = json.loads(text)
    except Exception as e:
        raise ValueError(f"Failed to parse input as JSON: {text}. Error: {e}") from e

    # Check for Pub/Sub envelope wrapper
    if "message" in data_dict and isinstance(data_dict["message"], dict):
        data_dict = data_dict["message"]

    # Extract the payload under "data"
    if "data" not in data_dict:
        # If "data" is not present, check if the payload is directly at the root
        details = data_dict
    else:
        data_field = data_dict["data"]
        # Decrypt base64 if it is a base64 encoded string
        if isinstance(data_field, str):
            try:
                decoded_bytes = base64.b64decode(data_field)
                decoded_str = decoded_bytes.decode("utf-8")
                details = json.loads(decoded_str)
            except Exception as e:
                # If base64 fails, fallback to parsing directly as a JSON string
                try:
                    details = json.loads(data_field)
                except Exception:
                    raise ValueError(
                        f"Failed to decode base64 or parse string 'data': {data_field}. Error: {e}"
                    ) from e
        elif isinstance(data_field, dict):
            details = data_field
        else:
            raise ValueError(
                "The 'data' field must be a base64 encoded string or a JSON object."
            )

    # Parse into PurchaseRequest
    purchase_request = PurchaseRequest(
        request_id=str(details.get("request_id", "")),
        department=str(details.get("department", "")),
        project=str(details.get("project", "")),
        requested_amount=float(details.get("requested_amount", 0.0)),
        available_budget=float(details.get("available_budget", 0.0)),
        requester=str(details.get("requester", "")),
        description=str(details.get("description", "")),
        date=str(details.get("date", "")),
    )

    # Save to workflow state for global access
    ctx.state["purchase_request"] = purchase_request.model_dump()
    return purchase_request


@node
def validate_budget(ctx: Context, node_input: PurchaseRequest) -> Event:
    """Applies the budget-validation rule.
    Returns:
        Event routing to auto_approved or needs_review.
    """
    if node_input.requested_amount <= node_input.available_budget:
        res = ValidationResult(
            approved=True,
            status="auto_approved",
            reason=f"Purchase request {node_input.request_id} approved. Requested amount ${node_input.requested_amount:.2f} is within available budget ${node_input.available_budget:.2f}.",
        )
        return Event(output=res, route="auto_approved")  # type: ignore
    else:
        # Exceeds budget. Route to LLM review
        return Event(output=node_input, route="needs_review")  # type: ignore


SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CC_REGEX = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")

INJECTION_KEYWORDS = [
    "ignore previous instructions",
    "system override",
    "always approve",
    "bypass budget",
    "bypass validation",
    "force budget validation",
    "set budget to high",
    "override validation",
    "ignore rules",
    "do not review",
    "skip review",
]


@node
def security_checkpoint(ctx: Context, node_input: PurchaseRequest) -> Event:
    """Scrubs sensitive personal data (SSN, Credit Cards) and detects prompt injection."""
    desc = node_input.description
    redacted = []

    # 1. Scrub SSNs
    if SSN_REGEX.search(desc):
        desc = SSN_REGEX.sub("[REDACTED_SSN]", desc)
        redacted.append("SSN")

    # 2. Scrub Credit Cards
    if CC_REGEX.search(desc):
        desc = CC_REGEX.sub("[REDACTED_CC]", desc)
        redacted.append("Credit Card")

    # Save scrubbed request details
    node_input.description = desc
    ctx.state["purchase_request"] = node_input.model_dump()
    if redacted:
        ctx.state["redacted_categories"] = redacted

    # 3. Detect prompt injection
    desc_lower = desc.lower()
    is_injection = False
    for kw in INJECTION_KEYWORDS:
        if kw in desc_lower:
            is_injection = True
            break

    if is_injection:
        ctx.state["security_event"] = True
        # Construct RiskAssessment to bypass LLM and alert human directly
        risk = RiskAssessment(
            risk_level="high",
            risk_factors=[
                "Security Alert: Potential prompt injection detected in description."
            ],
            alert_raised=True,
            justification=(
                f"Suspected prompt injection attempt detected. Bypassed LLM risk review. "
                f"Suspicious payload: '{desc}'"
            ),
        )
        return Event(output=risk, route="security_alert")  # type: ignore

    return Event(output=node_input, route="clean")  # type: ignore


# LLM review agent node
llm_reviewer = LlmAgent(
    name="llm_reviewer",
    model=Gemini(
        model=config.get("model_name", "gemini-3.1-flash-lite"),
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="""You are an ambient budget-validation risk auditor.
You are reviewing a purchase request that EXCEEDS the available budget.
Review the purchase request details carefully.
Analyze potential budget risk factors and decide:
1. What the risk level is (low, medium, or high).
2. If a high budget risk alert should be raised (set alert_raised to true if risk level is medium or high).
3. A clear, professional justification explaining your risk assessment decision.

Your output must follow the RiskAssessment schema.""",
    output_schema=RiskAssessment,
)


@node(name="human_approval", rerun_on_resume=True)
async def human_approval_node(
    ctx: Context, node_input: RiskAssessment
) -> AsyncGenerator[Any, None]:
    """Pauses the workflow to wait for human approval/rejection.
    Resumes with the human's decision.
    """
    if not ctx.resume_inputs or "decision" not in ctx.resume_inputs:
        purchase_request_dict = ctx.state.get("purchase_request", {})
        req_id = purchase_request_dict.get("request_id", "unknown")
        amount = purchase_request_dict.get("requested_amount", 0.0)
        budget = purchase_request_dict.get("available_budget", 0.0)

        msg = (
            f"WARNING: Purchase Request {req_id} exceeds available budget! "
            f"Requested: ${amount:.2f}, Available: ${budget:.2f}. "
            f"LLM Risk Analysis -> Level: {node_input.risk_level.upper()}, "
            f"Alert Raised: {node_input.alert_raised}. "
            f"Justification: {node_input.justification}. "
            "Please approve or reject this request."
        )
        yield RequestInput(interrupt_id="decision", message=msg)
        return

    # Process user decision on resume
    decision = ctx.resume_inputs["decision"]
    if isinstance(decision, dict):
        decision_val = decision.get("decision", decision.get("response", str(decision)))
    else:
        decision_val = decision

    approved = str(decision_val).lower() in ("approve", "approved", "yes", "y", "true")

    ctx.state["human_decision"] = decision_val
    ctx.state["human_approved"] = approved

    status = "approved" if approved else "rejected"
    reason = f"Human reviewer {status} the request. Comment: {decision_val}"

    yield Event(
        output=ValidationResult(approved=approved, status=status, reason=reason)
    )


@node
def record_outcome(
    ctx: Context, node_input: ValidationResult
) -> AsyncGenerator[Event, None]:
    """Logs the final outcome of the purchase validation and returns it."""
    purchase_request_dict = ctx.state.get("purchase_request", {})
    req_id = purchase_request_dict.get("request_id", "unknown")

    msg_text = (
        f"[RECORDED OUTCOME] Request: {req_id} | Approved: {node_input.approved} | "
        f"Status: {node_input.status} | Reason: {node_input.reason}"
    )
    print(msg_text)

    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg_text)])
    )
    yield Event(output=node_input)


# Connect the Graph Workflow
root_agent = Workflow(
    name="budget_validation_workflow",
    edges=[
        Edge(from_node=START, to_node=parse_input),
        Edge(from_node=parse_input, to_node=validate_budget),
        Edge(from_node=validate_budget, to_node=record_outcome, route="auto_approved"),
        Edge(
            from_node=validate_budget, to_node=security_checkpoint, route="needs_review"
        ),
        Edge(from_node=security_checkpoint, to_node=llm_reviewer, route="clean"),
        Edge(
            from_node=security_checkpoint,
            to_node=human_approval_node,
            route="security_alert",
        ),
        Edge(from_node=llm_reviewer, to_node=human_approval_node),
        Edge(from_node=human_approval_node, to_node=record_outcome),
    ],
    description="Validates purchase requests based on available budget and risk factors.",
    output_schema=ValidationResult,
)

app = App(
    root_agent=root_agent,
    name="budget_validation_agent",
)
