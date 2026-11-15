.PHONY: install test train

install:
	python3 -m pip install -r requirements.txt
	python3 -m pip install --no-deps -e ".[dev]"

test:
	pytest -q

train:
	rul-train
