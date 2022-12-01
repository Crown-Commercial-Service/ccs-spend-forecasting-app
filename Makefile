unit_testing:
	poetry run python -m unittest discover -p '*_test.py' -s test

format:
	poetry run black .

format_check:
	poetry run black . --check

lint:
	poetry run pylint test