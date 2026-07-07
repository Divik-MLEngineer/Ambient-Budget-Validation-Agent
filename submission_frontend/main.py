import os
import json
import httpx
import google.auth
import google.auth.transport.requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService

# Initialize FastAPI
app = FastAPI(title="Budget Validation Manager Dashboard")

# Load configuration from environment
_, PROJECT_ID = google.auth.default()
if not PROJECT_ID:
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

AGENT_RUNTIME_ID = os.getenv("AGENT_RUNTIME_ID")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east1")

# Extract short engine ID for VertexAiSessionService
engine_id = AGENT_RUNTIME_ID.split("/")[-1] if AGENT_RUNTIME_ID else None

# Helper to resolve the full Reasoning Engine resource name
if AGENT_RUNTIME_ID and not AGENT_RUNTIME_ID.startswith("projects/"):
    full_engine_path = f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{AGENT_RUNTIME_ID}"
else:
    full_engine_path = AGENT_RUNTIME_ID

# Initialize session service
session_service = VertexAiSessionService(
    project=PROJECT_ID, location=LOCATION, agent_engine_id=engine_id
)


def get_gcp_token() -> str:
    """Helper to dynamically fetch and refresh GCP OAuth 2.0 access token."""
    credentials, _ = google.auth.default()
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    return credentials.token


@app.get("/api/pending")
async def list_pending_approvals():
    """Lists all sessions currently paused on a human approval step."""
    if not engine_id or not PROJECT_ID:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_CLOUD_PROJECT or AGENT_RUNTIME_ID environment variable is missing.",
        )

    try:
        # 1. List all active sessions on Agent Runtime
        sessions_res = await session_service.list_sessions(
            app_name="budget_validation_agent"
        )

        pending_list = []
        for session in sessions_res.sessions:
            # 2. Get full details of each session to access its events
            full_session = await session_service.get_session(
                app_name="budget_validation_agent",
                user_id=session.user_id,
                session_id=session.id,
            )

            if not full_session or not full_session.events:
                continue

            # Scan for adk_request_input calls and responses
            interrupts = {}
            responses = {}
            for event in full_session.events:
                content = getattr(event, "content", None)
                if not content:
                    continue
                parts = getattr(content, "parts", None)
                if not parts:
                    continue
                for part in parts:
                    # Check for function_call requesting input
                    fc = getattr(part, "function_call", None)
                    if fc and getattr(fc, "name", None) == "adk_request_input":
                        fc_id = getattr(fc, "id", None)
                        fc_args = getattr(fc, "args", {}) or {}
                        message = fc_args.get("message", "")
                        interrupts[fc_id] = message

                    # Check for function_response providing input
                    fr = getattr(part, "function_response", None)
                    if fr and getattr(fr, "name", None) == "adk_request_input":
                        fr_id = getattr(fr, "id", None)
                        responses[fr_id] = True

            # Identify unresolved interrupts
            for fc_id, message in interrupts.items():
                if fc_id not in responses:
                    # Extract purchase request payload from state
                    purchase_request = full_session.state.get("purchase_request")

                    # Fallback: Parse request details from first user message if state is empty
                    if not purchase_request and full_session.events:
                        for e in full_session.events:
                            if (
                                getattr(e, "author", None) == "user"
                                and e.content
                                and e.content.parts
                            ):
                                try:
                                    user_text = e.content.parts[0].text
                                    user_json = json.loads(user_text)
                                    if "data" in user_json:
                                        purchase_request = user_json["data"]
                                    else:
                                        purchase_request = user_json
                                    break
                                except Exception:
                                    pass

                    # Extract LLM risk assessment from events if available
                    llm_risk_info = None
                    for event in full_session.events:
                        if (
                            getattr(event, "author", None) == "llm_reviewer"
                            and event.content
                            and event.content.parts
                        ):
                            try:
                                text_data = event.content.parts[0].text
                                risk_data = json.loads(text_data)
                                if "risk_level" in risk_data:
                                    llm_risk_info = risk_data
                                    break
                            except Exception:
                                pass

                    pending_list.append(
                        {
                            "session_id": session.id,
                            "interrupt_id": fc_id,
                            "message": message,
                            "purchase_request": purchase_request,
                            "llm_risk": llm_risk_info,
                            "user_id": session.user_id,
                        }
                    )

        return pending_list
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch pending requests: {e}"
        )


@app.post("/api/action/{session_id}")
async def take_action(session_id: str, payload: dict):
    """Resumes the workflow in the specified session with the manager's decision."""
    if not full_engine_path:
        raise HTTPException(
            status_code=500, detail="AGENT_RUNTIME_ID environment variable is missing."
        )

    approved = payload.get("approved", True)
    interrupt_id = payload.get("interrupt_id", "decision")

    # Construct the resume payload following the exact schema
    resume_payload = {
        "class_method": "async_stream_query",
        "input": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": "adk_request_input",
                            "id": interrupt_id,
                            "response": {
                                "approved": approved,
                                "decision": "approve" if approved else "reject",
                            },
                        }
                    }
                ],
            },
            "session_id": session_id,
            "user_id": "default-user",  # Strictly set default-user to avoid ownership mismatch
        },
    }

    token = get_gcp_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/{full_engine_path}:streamQuery"

    final_result = None
    stream_events = []

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", url, headers=headers, json=resume_payload
            ) as response:
                if response.status_code != 200:
                    err_text = await response.aread()
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Failed to resume session on Agent Runtime: {err_text.decode()}",
                    )

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    stream_events.append(event)

                    # Extract the ValidationResult output
                    output = event.get("output")
                    if (
                        output
                        and isinstance(output, dict)
                        and ("approved" in output or "status" in output)
                    ):
                        final_result = output

        return {
            "status": "success",
            "session_id": session_id,
            "validation_result": final_result,
            "events": stream_events,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the manager dashboard web page styled with sleek glassmorphism."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Budget Manager Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: 'Outfit', sans-serif;
            background-color: #08090d;
            color: #f3f4f6;
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }
        /* Background Glows */
        .bg-glow-1 {
            position: absolute;
            top: -10%;
            left: -10%;
            width: 50vw;
            height: 50vw;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, rgba(0,0,0,0) 70%);
            z-index: -1;
            filter: blur(120px);
            pointer-events: none;
        }
        .bg-glow-2 {
            position: absolute;
            bottom: -10%;
            right: -10%;
            width: 50vw;
            height: 50vw;
            background: radial-gradient(circle, rgba(14, 165, 233, 0.12) 0%, rgba(0,0,0,0) 70%);
            z-index: -1;
            filter: blur(120px);
            pointer-events: none;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 3rem 2rem;
        }
        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 2rem;
        }
        .title-area h1 {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .title-area p {
            color: #9ca3af;
            margin-top: 0.5rem;
            font-size: 1rem;
        }
        .header-actions {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .status-badge {
            background: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.2);
            color: #10b981;
            padding: 0.5rem 1rem;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10b981;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        .btn-refresh {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #fff;
            padding: 0.65rem 1.25rem;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            font-family: inherit;
        }
        .btn-refresh:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }
        /* Grid Layout */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 2.5rem;
        }
        /* Glassmorphism Cards */
        .card {
            background: rgba(255, 255, 255, 0.01);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 24px;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #6366f1, #0ea5e9);
            opacity: 0;
            transition: opacity 0.3s;
            border-radius: 24px 24px 0 0;
        }
        .card:hover {
            transform: translateY(-6px);
            background: rgba(255, 255, 255, 0.03);
            border-color: rgba(255, 255, 255, 0.12);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
        }
        .card:hover::before {
            opacity: 1;
        }
        /* Card Elements */
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        .amount-display {
            font-size: 1.8rem;
            font-weight: 800;
            color: #f87171;
            letter-spacing: -0.02em;
        }
        .amount-sub {
            font-size: 0.85rem;
            color: #9ca3af;
            margin-top: 0.25rem;
        }
        .project-badge {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 0.4rem 0.8rem;
            border-radius: 10px;
            font-size: 0.8rem;
            font-weight: 600;
            color: #e5e7eb;
        }
        .info-list {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding: 1.2rem 0;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
        }
        .info-label {
            color: #6b7280;
        }
        .info-value {
            color: #d1d5db;
            font-weight: 500;
        }
        .description-box {
            font-size: 0.9rem;
            color: #9ca3af;
            line-height: 1.5;
            background: rgba(255, 255, 255, 0.01);
            padding: 0.85rem 1rem;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.02);
        }
        .risk-box {
            background: rgba(239, 68, 68, 0.04);
            border: 1px solid rgba(239, 68, 68, 0.15);
            border-radius: 16px;
            padding: 1.1rem;
        }
        .risk-title {
            font-size: 0.8rem;
            font-weight: 700;
            color: #fca5a5;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .risk-desc {
            font-size: 0.85rem;
            color: #fecaca;
            line-height: 1.5;
        }
        /* Buttons */
        .actions-row {
            display: flex;
            gap: 1rem;
            margin-top: auto;
        }
        .btn {
            flex: 1;
            padding: 0.9rem;
            border-radius: 14px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
            border: none;
            font-family: inherit;
        }
        .btn-approve {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: #fff;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.15);
        }
        .btn-approve:hover {
            box-shadow: 0 6px 22px rgba(16, 185, 129, 0.35);
            transform: translateY(-2px);
        }
        .btn-reject {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: #fff;
            box-shadow: 0 4px 14px rgba(239, 68, 68, 0.15);
        }
        .btn-reject:hover {
            box-shadow: 0 6px 22px rgba(239, 68, 68, 0.35);
            transform: translateY(-2px);
        }
        .btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none !important;
            box-shadow: none !important;
        }
        .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
            display: none;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        /* Drawer */
        .drawer {
            position: fixed;
            top: 0;
            right: -480px;
            width: 440px;
            height: 100vh;
            background: rgba(10, 11, 16, 0.94);
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            border-left: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: -20px 0 50px rgba(0, 0, 0, 0.6);
            z-index: 1000;
            padding: 3rem 2.5rem;
            box-sizing: border-box;
            transition: right 0.4s cubic-bezier(0.25, 1, 0.5, 1);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        .drawer.open {
            right: 0;
        }
        .drawer-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(4px);
            z-index: 999;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s;
        }
        .drawer-overlay.open {
            opacity: 1;
            pointer-events: auto;
        }
        .drawer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 1.2rem;
        }
        .drawer-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #fff;
        }
        .btn-close {
            background: none;
            border: none;
            color: #9ca3af;
            cursor: pointer;
            font-size: 1.6rem;
            transition: color 0.2s;
        }
        .btn-close:hover {
            color: #fff;
        }
        .drawer-content {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        .outcome-banner {
            padding: 1.2rem;
            border-radius: 16px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .outcome-approved {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            color: #34d399;
        }
        .outcome-rejected {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: #f87171;
        }
        .json-block {
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1.25rem;
            font-family: monospace;
            white-space: pre-wrap;
            overflow-x: auto;
            color: #38bdf8;
            font-size: 0.85rem;
            line-height: 1.5;
        }
        /* Empty State */
        .empty-state {
            background: rgba(255, 255, 255, 0.01);
            border: 1px dashed rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 6rem 2rem;
            text-align: center;
            grid-column: 1 / -1;
            backdrop-filter: blur(10px);
        }
        .empty-icon {
            font-size: 3.5rem;
            color: #34d399;
            margin-bottom: 1.5rem;
        }
        .empty-state h3 {
            font-size: 1.6rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }
        .empty-state p {
            color: #9ca3af;
        }
        /* Loading Skeleton */
        .loading-skeleton {
            background: linear-gradient(90deg, rgba(255,255,255,0.02) 25%, rgba(255,255,255,0.06) 50%, rgba(255,255,255,0.02) 75%);
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
        }
        @keyframes loading {
            to { background-position: -200% 0; }
        }
    </style>
</head>
<body>
    <div class="bg-glow-1"></div>
    <div class="bg-glow-2"></div>

    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="title-area">
                <h1>Validation Pipeline</h1>
                <p>Ambient budget validation workflow approvals & escalations</p>
            </div>
            <div class="header-actions">
                <div class="status-badge">
                    <span class="status-dot"></span>
                    <span>Reasoning Engine Connected</span>
                </div>
                <button class="btn-refresh" onclick="fetchPendingRequests()">Refresh</button>
            </div>
        </header>

        <!-- Main Content -->
        <main id="dashboard-grid" class="grid">
            <!-- Rendered by JavaScript -->
        </main>
    </div>

    <!-- Drawer Component -->
    <div id="drawer-overlay" class="drawer-overlay" onclick="closeDrawer()"></div>
    <div id="side-drawer" class="drawer">
        <div class="drawer-header">
            <h2 class="drawer-title">Validation Complete</h2>
            <button class="btn-close" onclick="closeDrawer()">&times;</button>
        </div>
        <div class="drawer-content">
            <div id="outcome-banner" class="outcome-banner"></div>
            <p style="margin-top: 1rem; color: #9ca3af; font-weight: 500;">Review Outcome Payload:</p>
            <pre id="json-result" class="json-block"></pre>
        </div>
    </div>

    <script>
        // Fetch and Render pending requests
        async function fetchPendingRequests() {
            const grid = document.getElementById("dashboard-grid");
            // Show loading skeleton
            grid.innerHTML = `
                <div class="card loading-skeleton" style="height: 400px; opacity: 0.6;"></div>
                <div class="card loading-skeleton" style="height: 400px; opacity: 0.4;"></div>
                <div class="card loading-skeleton" style="height: 400px; opacity: 0.2;"></div>
            `;

            try {
                const response = await fetch("/api/pending");
                const data = await response.json();
                
                if (!Array.isArray(data) || data.length === 0) {
                    renderEmptyState();
                    return;
                }
                
                grid.innerHTML = "";
                data.forEach(item => {
                    const card = createCard(item);
                    grid.appendChild(card);
                });
            } catch (err) {
                grid.innerHTML = `
                    <div class="empty-state" style="border-color: rgba(239, 68, 68, 0.2);">
                        <div class="empty-icon" style="color: #ef4444;">⚠️</div>
                        <h3>Failed to Load Requests</h3>
                        <p style="color: #fca5a5;">${err.message || err}</p>
                    </div>
                `;
            }
        }

        function renderEmptyState() {
            const grid = document.getElementById("dashboard-grid");
            grid.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">✓</div>
                    <h3>Pipeline Clear</h3>
                    <p>No purchase requests are currently pending manager review.</p>
                </div>
            `;
        }

        function createCard(item) {
            const pr = item.purchase_request || {};
            const risk = item.llm_risk || {};
            
            const reqAmount = pr.requested_amount || 0;
            const availBudget = pr.available_budget || 0;
            const overage = reqAmount - availBudget;
            
            const card = document.createElement("div");
            card.className = "card";
            card.id = `card-${item.session_id}`;
            
            // Format numbers
            const fmt = (num) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(num);

            card.innerHTML = `
                <div class="card-header">
                    <div>
                        <div class="amount-display">${fmt(reqAmount)}</div>
                        <div class="amount-sub">${fmt(overage)} over budget</div>
                    </div>
                    <span class="project-badge">${pr.project || "Unknown Project"}</span>
                </div>
                
                <div class="info-list">
                    <div class="info-row">
                        <span class="info-label">Request ID</span>
                        <span class="info-value">${pr.request_id || "N/A"}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Requester</span>
                        <span class="info-value">${pr.requester || "N/A"}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Department</span>
                        <span class="info-value">${pr.department || "N/A"}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Date Submitted</span>
                        <span class="info-value">${pr.date || "N/A"}</span>
                    </div>
                </div>

                <div class="description-box">
                    <span style="font-weight: 600; color: #d1d5db; display: block; margin-bottom: 0.25rem;">Description</span>
                    ${pr.description || "No description provided."}
                </div>

                <div class="risk-box">
                    <div class="risk-title">
                        ⚠️ LLM Risk Review Alert
                    </div>
                    <p class="risk-desc">${risk.justification || item.message || "High budget risk factors detected. Escalated for human override."}</p>
                </div>

                <div class="actions-row">
                    <button class="btn btn-reject" onclick="handleDecision('${item.session_id}', '${item.interrupt_id}', false, this)">
                        <span class="spinner"></span>
                        <span>Reject</span>
                    </button>
                    <button class="btn btn-approve" onclick="handleDecision('${item.session_id}', '${item.interrupt_id}', true, this)">
                        <span class="spinner"></span>
                        <span>Approve</span>
                    </button>
                </div>
            `;
            return card;
        }

        // Action Decision handling
        async function handleDecision(sessionId, interruptId, approved, btnEl) {
            const card = document.getElementById(`card-${sessionId}`);
            const buttons = card.querySelectorAll(".btn");
            const spinner = btnEl.querySelector(".spinner");
            
            // Disable buttons and show spinner
            buttons.forEach(b => b.disabled = true);
            spinner.style.display = "inline-block";

            try {
                const response = await fetch(`/api/action/${sessionId}`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        approved: approved,
                        interrupt_id: interruptId
                    })
                });
                
                const data = await response.json();
                
                if (response.status === 200 && data.status === "success") {
                    showValidationResult(data.validation_result, approved);
                    // Remove card with animation
                    card.style.opacity = "0";
                    card.style.transform = "scale(0.9)";
                    setTimeout(() => {
                        card.remove();
                        // Check if dashboard empty
                        if (document.querySelectorAll(".card").length === 0) {
                            renderEmptyState();
                        }
                    }, 400);
                } else {
                    alert(`Error resuming session: ${data.detail || "Unknown error"}`);
                    // Re-enable buttons
                    buttons.forEach(b => b.disabled = false);
                    spinner.style.display = "none";
                }
            } catch (err) {
                alert(`Error: ${err.message || err}`);
                buttons.forEach(b => b.disabled = false);
                spinner.style.display = "none";
            }
        }

        // Drawer Handling
        function showValidationResult(result, approved) {
            const overlay = document.getElementById("drawer-overlay");
            const drawer = document.getElementById("side-drawer");
            const banner = document.getElementById("outcome-banner");
            const jsonBox = document.getElementById("json-result");

            if (approved) {
                banner.className = "outcome-banner outcome-approved";
                banner.innerHTML = "✓ Request Approved Successfully";
            } else {
                banner.className = "outcome-banner outcome-rejected";
                banner.innerHTML = "✕ Request Rejected & Closed";
            }

            jsonBox.textContent = JSON.stringify(result || { "status": approved ? "approved" : "rejected", "reason": "Manually validation override recorded." }, null, 2);

            overlay.classList.add("open");
            drawer.classList.add("open");
        }

        function closeDrawer() {
            document.getElementById("drawer-overlay").classList.remove("open");
            document.getElementById("side-drawer").classList.remove("open");
        }

        // Init
        document.addEventListener("DOMContentLoaded", fetchPendingRequests);
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
