.PHONY: install data test train serve

install:
	python3 -m pip install -r requirements.txt
	python3 -m pip install --no-deps -e ".[dev]"

data:
	rul-fetch

test:
	pytest -q

train:
	rul-train

serve:
	uvicorn engine_rul.service:app --port 8000
