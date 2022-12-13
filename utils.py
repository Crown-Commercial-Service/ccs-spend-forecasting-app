import logging.config
from configparser import ConfigParser

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection

config = ConfigParser()
config.read("config.ini")

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("ccsForecast")

DB_DIALECT = config.get("database", "dialect")
DB_SERVER = config.get("database", "server")
DB_PORT = config.get("database", "port")
DB_NAME = config.get("database", "name")
DB_USER = config.get("database", "user")
DB_PASSWORD = config.get("database", "password")
DB_DRIVER = config.get("database", "driver")
DB_POOL_SIZE = int(config.get("database", "pool_size"))
PY_DB_URL = f"{DB_DIALECT}://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}:{DB_PORT}/{DB_NAME}?driver={DB_DRIVER}"
JDBC_URL = (
    f"jdbc:sqlserver://{DB_SERVER}:{DB_PORT};database={DB_NAME};"
    + "encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;"
)
DB_ENGINE = create_engine(PY_DB_URL, pool_size=DB_POOL_SIZE)
DB_PROPERTIES_SPARK = {"user": DB_USER, "password": DB_PASSWORD}


def get_logger() -> logging.Logger:
    """Returns logger to be used in this app

    Returns:
        logger

    """

    return logger


def get_database_connection() -> Connection:
    """Call this method to get database connection.
    Returns:
        sqlalchemy connection object to query database.
    """

    return DB_ENGINE.connect()
