help:
	@grep -E '^[a-zA-Z_-]+:.*?## .$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:
	python -m venv .venv
	.venv\Scripts\python.exe -m pip install -e .[engines,dev]

run:
	.venv\Scripts\python.exe -m uvicorn vio.main:app --host 127.0.0.1 --port 8000

test:
	.venv\Scripts\python.exe -m pytest tests/ -v

clean:
	rm -rf __pycache__ .pytest_cache
	rm -f output*.wav

.PHONY: help install run test clean