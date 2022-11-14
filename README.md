# ccs-spend-forecasting-app
Application repository for Data Insights Spend Forecasting Project

## About
The goal of this project is to forecast the spend. <more to come.>

### Dependency management
Dependency is managed by [Poetry](https://python-poetry.org) 

### Usage:
Below are steps describe how to get started:
1. Checkout the project from the git.
2. Make a copy of the "file config.ini.template" and rename it to "config.ini".
3. Edit the file "config.ini" and add required details like username and password. **_Never checkin this file in git_**.
4. Create a folder called "logs" at the root of the application that will contain the application logs. Alternatively 
   you can edit the "logging.conf" file and edit the line:

   _args=("logs/app.log", 5242880, 3)_.

   There are the arguments as explained below:
   1. Log location: First argument defines the log location. By default, it is a folder call logs and contains log 
      files with name app.log.
   2. Log file size: Second parameter describes the size of log file in bytes. By default, it is 5MB.
   3. Number of log files: Third parameter describes number of log files before it is rolled. By default, it is 3.

