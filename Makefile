.PHONY: install test lint api dashboard run

install:
	python -m pip install -r requirements-dev.txt
	python -m pip install -e .

test:
	python -m pytest -q

lint:
	ruff check .

api:
	python -m uvicorn recommender_drift_monitor.api:app --app-dir src --reload --host 127.0.0.1 --port 8000

dashboard:
	python -m streamlit run dashboard.py

run:
	./run_local.sh
