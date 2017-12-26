import s3fs
import pyarrow.parquet as pq
import pandas as pd

s3 = s3fs.S3FileSystem()
dataset = pq.ParquetDataset('s3://kartees-cloud-collection/parquet-structured/sample_1506623581.parquet', filesystem=s3)

df = dataset.read_pandas()
pandas_data = df.to_pandas()
pandas_data.to_csv("spark-structured-sample.csv")

