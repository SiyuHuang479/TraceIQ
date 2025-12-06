from matplotlib.pylab import median
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, median



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
print(f"Original data count: {df.count()}")
print(f"Cleaned data count: {df_clean.count()}")






# ------------------ get insights ------------------
# scheduling delays in terms of CN and HN
df_delays = (
    df_clean
    .withColumn("sched_delay", col("scheduled_time") - col("creation_time"))   # calculate scheduling delay
    .filter(col("sched_delay").isNotNull() & (col("sched_delay") >= 0))         # filter out null or negative delays
)

# median and average scheduling delays
delay_stats = (
    df_delays
    .groupBy("role")
    .agg(avg("sched_delay").alias("avg_sched_delay"),
         median("sched_delay").alias("median_sched_delay"))
    .orderBy("role")
)

# show delay stats
print("HN vs CN Scheduling Delays:")
delay_stats.show()



# stop spark session
spark.stop()