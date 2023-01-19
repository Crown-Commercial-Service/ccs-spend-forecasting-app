unit_testing:
	poetry run pytest

format:
	poetry run black .

format_check:
	poetry run black . --check

lint:
	poetry run pylint pipeline test

pipeline_fill_missing_months:
	poetry run python -m pipeline.jobs.fill_missing_months

pipeline_get_active_combinations:
	poetry run python -m pipeline.jobs.get_active_combinations

pipeline_run_forecast:
	poetry run python -m pipeline.jobs.run_forecast_and_output_to_blob