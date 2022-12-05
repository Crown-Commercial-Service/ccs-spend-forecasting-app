from unittest import TestCase

from pyspark.sql import SparkSession


class ReusableSparkTestCase(TestCase):
    spark: SparkSession = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = (
            SparkSession.builder.config("spark.sql.debug.maxToStringFields", "10000")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
            .config("spark.sql.execution.arrow.pyspark.selfDestruct.enabled", "true")
            .config("spark.sql.shuffle.partitions", "7")
            .config("spark.driver.memory", "12g")
            .master("local[*]")
            .appName("SparkLocalTestCase")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()
