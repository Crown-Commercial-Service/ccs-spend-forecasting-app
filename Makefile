unit_testing:
	poetry run pytest

format:
	poetry run black .

format_check:
	poetry run black . --check

lint:
	poetry run pylint pipeline test