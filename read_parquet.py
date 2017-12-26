from pyspark.sql import SQLContext

spark = SparkSession.builder.getOrCreate()
hconf = spark.sparkContext._jsc.hadoopConfiguration()

hconf.set("fs.s3a.access.key", "AKIAJ5ARNY3UNOMLQ5JA")  
hconf.set("fs.s3a.secret.key", "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")

spark = SparkSession.builder.getOrCreate()

new_data = sqlContext.read.parquet("s3a://kartees-cloud-collection/parquet-structured/sample_1506626271.parquet")

new_data.show()