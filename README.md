# CCS Spend Forecasting

This is main repo for CCS spend forecasting project.

## About

The goal of this application is to perform financial forecasting for CCS. Currently it does monthly forecast for each
category (aka group) and market sector (aka segment). Though this can be changed to sub-category (aka category) with
subsequent change in data preparation and model training.
In CCS few terms are used interchangeably, below table list those:

| Entity referred in Code | Other Name |
|-------------------------|------------|
| Category                | Group      |
| Sub-Category            | Category   |
| Market Sector           | Segment    |

### Requirements

For the local development you need following libraries installed on your local machine:

- Java
- Python
- Poetry


Note: We have used Python version 3.9, so we recommend using that or above.
#### Java installation

In order to install Java on a Mac, the easiest way is to use Homebrew. Run `brew install openjdk`, and create a symlink
to allow your machine to locate the JDK:

```
sudo ln -sfn /opt/homebrew/opt/openjdk/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk
```

Alternatively you can also install from https://adoptopenjdk.net

Java version: Any version with Java 8 or above.

### Dependency management

Dependency is managed by [Poetry](https://python-poetry.org)

### Usage:

Below are steps describe how to get started on your local development environment. You can develop and run this project
from you local IDE. Below are the steps of how to run it on you local machine.

#### Configuration

Below are the steps for configuring project on your local machine:

1. Checkout the project from the git repo.
2. Run `poetry install` to install all required packages
3. Run `poetry run pre-commit install` to ensure that formatting takes place locally on every commit
4. Make a copy of the "file config.ini.template" and rename it to "config.ini".
5. Edit the file "config.ini" and add required details like username and password. **_Never checkin this file in git as
   this is a public repo_**.
6. Create a folder called "logs" at the root of the application that will contain the application logs. Alternatively
   you can edit the "logging.conf" file and edit the line:

   _args=("logs/app.log", 5242880, 3)_.

   There are the arguments as explained below:
    1. Log location: First argument defines the log location. By default, it is a folder call logs and contains log
       files with name app.log.
    2. Log file size: Second parameter describes the size of log file in bytes. By default, it is 5MB.
    3. Number of log files: Third parameter describes number of log files before it is rolled. By default, it is 3.


#### Exploratory Data Analysis

Below are the steps for running and performing data analysis on your local machine:

1. Explorator data analysis (EDA) is at [data_analysis.py](eda/data_analysis.py). To run EDA you have two options:
    1. Get the data by querying the database: For this you must have credentials and should be able to connect to the
       database. After you get the data you have to insert the missing months i.e. months for which there is no entry.
       This method is **slow** and requires effort.
    2. Get the data file from Azure: The ADF in azure prepares the data needed for the analysis and takes care of adding
       missing months. This data is saved in the parquet format. You can get this file and can perform EDA on this file
       which is much easier. For more information please see:
       1. https://github.com/Crown-Commercial-Service/ccs-spend-forecasting-infra
       2. https://github.com/Crown-Commercial-Service/ccs-spend-forecasting-adf
2. How to run [data_analysis.py](eda/data_analysis.py): We highly recommend to run the code from your favourite IDE as 
   it the easiest way play you get all the benefit of the IDE, however if you wish to run the code from command line use 
   below command:
    ```shell
   export PYTHONPATH=$PYTHONPATH:/Path/to/the/ccs-spend-forecasting-app
   python3 eda/data_analysis.py --local_data_path=/Path/of/the/folder/where/you/saved/the/parquet_files
    ```
   
3. To select a particular category and sector pass the index to the function get_category_sector(index) which contains
   the list of all category sector combination.
4. To run a particular model pass the flag run=True to that function. For example if you want to run SARIMA model:
   ```
    model_sarima(prepared_df, category=category, sector=sector, run=True)
    ```
5. Currently, the best hyperparameters are selected based on lowest AIC score is done to make the process fast. Ideally it 
   should be analysed qualitatively by the data scientist for each category & sector (there are over 150 different 
   combination) before finalising the best value.


#### How to run data pipeline

Below are the steps to run the data pipeline

1. TODO: Write about pipeline

### Testing
The unit test of this repo is located in `test` directory. You can use the below command in a terminal to run the full \
test suite:

```shell
python3 -m unittest discover -p '*_test.py' -s test
```