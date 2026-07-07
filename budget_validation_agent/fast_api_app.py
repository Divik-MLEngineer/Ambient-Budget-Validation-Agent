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

import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator

import google.auth
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.cloud import logging as google_cloud_logging
from google.genai import types

from budget_validation_agent.app_utils import services
from budget_validation_agent.app_utils.a2a import attach_a2a_routes
from budget_validation_agent.app_utils.reasoning_engine_adapter import (
    attach_reasoning_engine_routes,
)
from budget_validation_agent.app_utils.telemetry import setup_telemetry
from budget_validation_agent.app_utils.typing import Feedback

load_dotenv()
setup_telemetry()
_, project_id = google.auth.default()

logging.basicConfig(level=logging.INFO)
console_logger = logging.getLogger("budget_validation_agent")

logging_client = google_cloud_logging.Client()
logger = logging_client.logger(__name__)
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from budget_validation_agent.agent import app as adk_app
    from budget_validation_agent.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=False,
    lifespan=lifespan,
)
app.title = "ambient-budget-val-agent"
app.description = "API for interacting with the Agent ambient-budget-val-agent"

attach_reasoning_engine_routes(app)


@app.post("/")
async def handle_pubsub_trigger(request: Request):
    """Handles Pub/Sub push subscription trigger messages."""
    try:
        body = await request.json()
    except Exception as e:
        console_logger.error(f"Failed to parse request JSON: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from e

    console_logger.info(f"Received Pub/Sub event: {json.dumps(body)}")

    # Extract and normalize subscription name
    subscription_path = body.get("subscription")
    if not subscription_path:
        subscription_name = "local-trigger"
    else:
        # Normalize: 'projects/myproject/subscriptions/mysubscription' -> 'mysubscription'
        subscription_name = subscription_path.split("/")[-1]

    # Check for pub/sub message envelope
    message_data = body.get("message")
    if not message_data or not isinstance(message_data, dict):
        message_data = body

    # Run the workflow
    runner = request.app.state.runner
    adk_app_name = request.app.state.agent_app_name

    # Create a unique session for this Pub/Sub run
    session = await runner.session_service.create_session(
        app_name=adk_app_name, user_id=subscription_name
    )

    user_message = types.Content(
        role="user", parts=[types.Part.from_text(text=json.dumps(message_data))]
    )

    console_logger.info(
        f"Starting workflow run for subscription '{subscription_name}' in session '{session.id}'"
    )

    events = []
    async for event in runner.run_async(
        user_id=subscription_name,
        session_id=session.id,
        new_message=user_message,
    ):
        events.append(event)

    interrupted = any(bool(getattr(e, "long_running_tool_ids", None)) for e in events)

    final_output = None
    if not interrupted:
        for e in reversed(events):
            if e.output is not None:
                final_output = e.output
                break

    response_data = {
        "status": "interrupted" if interrupted else "completed",
        "session_id": session.id,
        "subscription": subscription_name,
    }

    if final_output:
        if hasattr(final_output, "model_dump"):
            response_data["output"] = final_output.model_dump()
        else:
            response_data["output"] = final_output

    console_logger.info(
        f"Workflow execution finished. Status: {response_data['status']}"
    )
    return response_data


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.log_struct(feedback.model_dump(), severity="INFO")
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
