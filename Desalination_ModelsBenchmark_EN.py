# Databricks notebook source
# Databricks Notebook: Compare Linear Regression, Random Forest, and Gradient Boosting Models

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
import pyspark.sql.functions as F

# Create Spark session
spark = SparkSession.builder.appName("Desalination_ML_Compare_Models").getOrCreate()

# Load the dataset from CSV (dbfs)
mock_data_path = "dbfs:/FileStore/input/mock_desalination_data.csv"  # Adjust the path if necessary
df = spark.read.csv(mock_data_path, header=True, inferSchema=True)

# Show the first few rows
df.show(5)

# -------------------- Data Cleaning -------------------- #
# Check for null values
df = df.dropna()

# View the dataset schema
df.printSchema()

# -------------------- Binning (Outside of Pipeline) -------------------- #
# Check if the column 'Feed_TDS_binned' already exists and remove it if necessary
if 'Feed_TDS_binned' in df.columns:
    df = df.drop('Feed_TDS_binned')

# Bin 'Feed TDS (ppm)' into 5 intervals using 'Bucketizer'
from pyspark.ml.feature import Bucketizer

splits = [0, 200, 400, 600, 800, 1000]  # Define the intervals for binning
bucketizer = Bucketizer(splits=splits, inputCol="Feed TDS (ppm)", outputCol="Feed_TDS_binned")
df = bucketizer.setHandleInvalid("skip").transform(df)

# -------------------- OneHotEncoder -------------------- #
# Convert the binned column into OneHotEncoded format
indexer = StringIndexer(inputCol="Feed_TDS_binned", outputCol="Feed_TDS_index")
df = indexer.fit(df).transform(df)

encoder = OneHotEncoder(inputCol="Feed_TDS_index", outputCol="Feed_TDS_ohe")
df = encoder.fit(df).transform(df)

# -------------------- New Columns: Energy Consumption and Costs -------------------- #
# Add a constant for kWh calculation (adjust as needed)
energy_constant = 0.05  # Adjustable constant

# Calculate kWh based on flowrate and pressure drop
df = df.withColumn('Energy (kWh)', F.col('Feed Flowrate (m3/h)') * F.col('Pressure Drop (bar)') * energy_constant)

# Add a constant for price per kWh (adjust as needed)
price_per_kwh = 0.15  # Price per kWh in €

# Calculate the cost (€)
df = df.withColumn('Cost (€)', F.col('Energy (kWh)') * price_per_kwh)

# Show the new columns
df.select("Feed Flowrate (m3/h)", "Pressure Drop (bar)", "Energy (kWh)", "Cost (€)").show(5)

# -------------------- VectorAssembler -------------------- #
# Include the new features in the feature vector
feature_columns = ['Feed Flowrate (m3/h)', 'Feed TDS (ppm)', 'Feed Salinity (g/L)', 'Feed Temperature (°C)', 
                   'Permeate Flowrate (m3/h)', 'Permeate TDS (ppm)', 'Recovery (%)', 'Flux (LMH)', 'Feed_TDS_ohe',
                   'Energy (kWh)', 'Cost (€)']  # Add the new columns

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df = assembler.transform(df)

# -------------------- Train-Test Split -------------------- #
# Split the data into training (80%) and testing (20%)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# -------------------- Evaluator (Used for all models) -------------------- #
evaluator_rmse = RegressionEvaluator(labelCol="Pressure Drop (bar)", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="Pressure Drop (bar)", predictionCol="prediction", metricName="r2")

import time

# -------------------- Model 1: Linear Regression -------------------- #
print("==> Training Linear Regression")
start_time = time.time()  # Start timing
lr = LinearRegression(featuresCol="features", labelCol="Pressure Drop (bar)")
lr_model = lr.fit(train_data)
end_time = time.time()  # End timing
lr_training_time = end_time - start_time
print(f"Linear Regression training time: {lr_training_time:.2f} seconds")

# Predictions and Evaluation
lr_predictions = lr_model.transform(test_data)
lr_rmse = evaluator_rmse.evaluate(lr_predictions)
lr_r2 = evaluator_r2.evaluate(lr_predictions)

print(f"Linear Regression - Root Mean Squared Error (RMSE): {lr_rmse}")
print(f"Linear Regression - R^2 Score: {lr_r2}")


# -------------------- Model 2: Random Forest Regressor -------------------- #
print("\n==> Training Random Forest Regressor")
start_time = time.time()  # Start timing
rf = RandomForestRegressor(featuresCol="features", labelCol="Pressure Drop (bar)", numTrees=50)
rf_model = rf.fit(train_data)
end_time = time.time()  # End timing
rf_training_time = end_time - start_time
print(f"Random Forest training time: {rf_training_time:.2f} seconds")

# Predictions and Evaluation
rf_predictions = rf_model.transform(test_data)
rf_rmse = evaluator_rmse.evaluate(rf_predictions)
rf_r2 = evaluator_r2.evaluate(rf_predictions)

print(f"Random Forest - Root Mean Squared Error (RMSE): {rf_rmse}")
print(f"Random Forest - R^2 Score: {rf_r2}")


# -------------------- Model 3: Gradient Boosting Regressor -------------------- #
print("\n==> Training Gradient Boosting Regressor")
start_time = time.time()  # Start timing
gbt = GBTRegressor(featuresCol="features", labelCol="Pressure Drop (bar)", maxIter=100)
gbt_model = gbt.fit(train_data)
end_time = time.time()  # End timing
gbt_training_time = end_time - start_time
print(f"Gradient Boosting training time: {gbt_training_time:.2f} seconds")

# Predictions and Evaluation
gbt_predictions = gbt_model.transform(test_data)
gbt_rmse = evaluator_rmse.evaluate(gbt_predictions)
gbt_r2 = evaluator_r2.evaluate(gbt_predictions)

print(f"Gradient Boosting - Root Mean Squared Error (RMSE): {gbt_rmse}")
print(f"Gradient Boosting - R^2 Score: {gbt_r2}")

# -------------------- Training Time Comparison -------------------- #
print("\n==> Training Time Comparison")
print(f"Linear Regression training time: {lr_training_time:.2f} seconds")
print(f"Random Forest training time: {rf_training_time:.2f} seconds")
print(f"Gradient Boosting training time: {gbt_training_time:.2f} seconds")

