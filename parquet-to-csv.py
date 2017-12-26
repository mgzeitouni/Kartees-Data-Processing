import s3fs
import pyarrow.parquet as pq

s3 = s3fs.S3FileSystem()
dataset = pq.ParquetDataset('s3://kartees-ai/price-through-time/new-bucket_2017_11_09_18_36_35.parquet', filesystem=s3)

df = dataset.read_pandas()
pandas_data = df.to_pandas()
pandas_data.to_csv("spark-structured.csv")

