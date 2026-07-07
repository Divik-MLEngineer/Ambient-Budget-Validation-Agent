import asyncio
import json
import os

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from budget_validation_agent.agent import root_agent

load_dotenv()


def clean_event(event):
    author = event.author or "system"
    content = None
    if event.content:
        content = json.loads(event.content.model_dump_json(exclude_none=True))
    elif event.output:
        output_str = str(event.output)
        node_name = (
            event.node_info.path.split("/")[-1]
            if event.node_info and event.node_info.path
            else "Node"
        )
        content = {
            "role": "model",
            "parts": [{"text": f"[{node_name} Output] {output_str}"}],
        }
    else:
        if event.long_running_tool_ids:
            content = {
                "role": "model",
                "parts": [
                    {
                        "text": f"[Human Approval Prompt] Active interrupt IDs: {list(event.long_running_tool_ids)}"
                    }
                ],
            }

    if not content:
        return None

    return {"author": author, "content": content}


async def main():
    dataset_path = "tests/eval/datasets/basic-dataset.json"
    output_path = "artifacts/traces/generated_traces.json"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(dataset_path) as f:
        dataset = json.load(f)

    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        session_service=session_service,
        app_name="budget_validation_agent",
    )

    eval_cases = []

    for case in dataset["eval_cases"]:
        case_id = case["eval_case_id"]
        print(f"Running scenario: {case_id}...")

        # Create a new session for this case
        session = await session_service.create_session(
            app_name="budget_validation_agent", user_id="eval-user"
        )

        turns = []
        turn_index = 0
        events_in_turn = []

        # 1. Start the workflow
        user_prompt_content = case["prompt"]
        events_in_turn.append({"author": "user", "content": user_prompt_content})

        is_interrupted = False
        final_validation_result = None

        async for event in runner.run_async(
            user_id="eval-user",
            session_id=session.id,
            new_message=types.Content(
                role=user_prompt_content.get("role", "user"),
                parts=[
                    types.Part.from_text(text=p["text"])
                    for p in user_prompt_content["parts"]
                ],
            ),
        ):
            if event.long_running_tool_ids:
                is_interrupted = True
            if event.output and hasattr(event.output, "approved"):
                final_validation_result = event.output
            e_dict = clean_event(event)
            if e_dict:
                events_in_turn.append(e_dict)

        turns.append({"turn_index": turn_index, "events": events_in_turn})

        # 2. Resume if interrupted
        if is_interrupted:
            decision = "reject" if "injection" in case_id else "approve"
            print(
                f"  Workflow paused for human approval. Automating decision: {decision}"
            )

            turn_index += 1
            events_in_turn = []
            decision_content = {"role": "user", "parts": [{"text": decision}]}
            events_in_turn.append({"author": "user", "content": decision_content})

            resume_message = types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="adk_request_input",
                            id="decision",
                            response={"decision": decision},
                        )
                    )
                ],
            )

            async for event in runner.run_async(
                user_id="eval-user",
                session_id=session.id,
                new_message=resume_message,
            ):
                if event.output and hasattr(event.output, "approved"):
                    final_validation_result = event.output
                e_dict = clean_event(event)
                if e_dict:
                    events_in_turn.append(e_dict)

            turns.append({"turn_index": turn_index, "events": events_in_turn})

        # Construct candidate responses list to satisfy SDK schema
        responses = []
        if final_validation_result:
            val_res_dict = json.loads(final_validation_result.model_dump_json())
            responses.append(
                {
                    "response": {
                        "role": "model",
                        "parts": [{"text": json.dumps(val_res_dict)}],
                    }
                }
            )
        else:
            responses.append(
                {
                    "response": {
                        "role": "model",
                        "parts": [
                            {
                                "text": "Workflow did not produce a final validation result."
                            }
                        ],
                    }
                }
            )

        eval_cases.append(
            {
                "eval_case_id": case_id,
                "prompt": case["prompt"],
                "responses": responses,
                "agent_data": {
                    "agents": {
                        "budget_validation_workflow": {
                            "agent_id": "budget_validation_workflow",
                            "agent_type": "Workflow",
                        }
                    },
                    "turns": turns,
                },
            }
        )

    output_data = {"eval_cases": eval_cases}
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(
        f"\nTrace generation complete. Saved {len(eval_cases)} traces to {output_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())
