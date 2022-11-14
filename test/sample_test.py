from base_test import ReusableSparkTestCase


class MySampleTest(ReusableSparkTestCase):

    def test_sample(self):
        columns = ["id"]
        data = [(1,), (2,), (3,)]
        df = self.spark.createDataFrame(data).toDF(*columns)
        self.assertEqual(3, df.count())
