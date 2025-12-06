from matplotlib.pylab import median
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, median, expr
import json 
import os 
import pandas as pd 
from sqlalchemy import create_engine

HIGH_PRECISION = 1000000


# ------------- initialization and loading data ----------------
# create spark session
spark = SparkSession.builder \
        .appName("DataLoader") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "localhost") \
        .getOrCreate()

# data path
data_path = "disaggregated_DLRM_trace.csv"

# load data
df = spark.read.csv(data_path, header=True, inferSchema=True)

# debugging print
#df.printSchema()
#df.show(5)



# ------------------ data cleaning ------------------
# check Requests <= Limits
request_limit_check = (
    (col("cpu_request") <= col("cpu_limit")) &
    (col("memory_request") <= col("memory_limit")) &
    (col("gpu_request") <= col("gpu_limit")) &
    (col("rdma_request") <= col("rdma_limit")) &
    (col("disk_request") <= col("disk_limit"))
)

# check creation_time <= scheduled_time
time_check_1 = (
    col("creation_time").isNull() |
    col("scheduled_time").isNull() |
    (col("creation_time") <= col("scheduled_time"))
)

# check scheduled_time <= deletion_time
time_check_2 = (
    col("scheduled_time").isNull() |
    col("deletion_time").isNull() |
    (col("scheduled_time") <= col("deletion_time"))
)

time_sequence_check = time_check_1 & time_check_2

# filter valid data
df_clean = df.filter(request_limit_check & time_sequence_check)

# debugging print
#print(f"Original data count: {df.count()}")
#print(f"Cleaned data count: {df_clean.count()}")






# ------------------ get insights ------------------
# scheduling delays in terms of CN and HN
valid_df_delays = (
    df_clean
    .withColumn("sched_delay", col("scheduled_time") - col("creation_time"))   # calculate scheduling delay
    .filter(col("sched_delay").isNotNull() & (col("sched_delay") >= 0))         # filter out null or negative delays
)

# debugging print
valid_delay_count = valid_df_delays.count()
print(f"Total rows in df_clean: {df_clean.count()}")
print(f"Total rows in valid_df_delays: {valid_delay_count}")

# median and average scheduling delays
delay_stats = (
    valid_df_delays
    .groupBy("role")
    .agg(avg("sched_delay").alias("avg_sched_delay"),
         expr("percentile_approx(sched_delay, 0.5)").alias("median_sched_delay"))
    .orderBy("role")
)

# show delay stats
print("HN vs CN Scheduling Delays:")
delay_stats.show()


PERCENTILES = [0.5, 0.90, 0.95, 0.99] # 中位数, P90, P95, P99

delay_percentiles = (
    valid_df_delays
    .groupBy("role")
    .agg(
        avg("sched_delay").alias("avg_sched_delay"),
        # 使用 percentile_approx 计算多个百分位数 (结果是一个数组)
        expr("percentile_approx(sched_delay, array(0.5, 0.90, 0.95, 0.99))").alias("delay_percentiles")
    )
    .orderBy("role")
)

print("HN vs CN 调度延迟百分位数 (P50, P90, P95, P99):")
delay_percentiles.show(truncate=False)



# ------------------ save results  ------------------
OUTPUT_PARQUET_PATH = "cluster_trace.parquet"
df_clean.write.mode("overwrite").parquet(OUTPUT_PARQUET_PATH)
print(f"Cleaned data saved to {OUTPUT_PARQUET_PATH}")







# stop spark session
spark.stop()