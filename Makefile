.PHONY: test
test:
	poetry env use ./.venv/bin/python
	poetry run pytest
