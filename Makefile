# Makefile for ambient-budget-val-agent

.PHONY: install run playground test lint generate-traces grade

install:
	uv sync

run:
	uv run uvicorn budget_validation_agent.fast_api_app:app --host 0.0.0.0 --port 8080

playground:
	uv run agents-cli playground

generate-traces:
	uv run python tests/eval/generate_traces.py

grade:
	uv run agents-cli eval grade --traces artifacts/traces/generated_traces.json --config tests/eval/eval_config.yaml
