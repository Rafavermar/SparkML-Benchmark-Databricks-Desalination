# Databricks notebook source
# Databricks Notebook: Sensitivity Analysis on Pressure Drop, kWh, and Cost

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor
import pyspark.sql.functions as F

# Create the Spark session
spark = SparkSession.builder.appName("Desalination_ML_Sensitivity_Analysis").getOrCreate()

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
# Check if the column Feed_TDS_binned already exists and remove it if necessary
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

# Calculate kWh based on pressure and flowrate
df = df.withColumn('Energy (kWh)', F.col('Feed Flowrate (m3/h)') * F.col('Pressure Drop (bar)') * energy_constant)

# Add a constant for the price per kWh (adjust as needed)
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

model_save_path = "dbfs:/FileStore/output/models/desalination_models/"

# -------------------- Gradient Boosting Regressor -------------------- #
print("==> Training Gradient Boosting Regressor")
gbt = GBTRegressor(featuresCol="features", labelCol="Pressure Drop (bar)", maxIter=100)
gbt_model = gbt.fit(train_data)
gbt_model.save(f"{model_save_path}/gradient_boosting_model_sensitivity")

# Predictions and Evaluation
gbt_predictions = gbt_model.transform(test_data)

# Evaluator to calculate metrics
evaluator_rmse = RegressionEvaluator(labelCol="Pressure Drop (bar)", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="Pressure Drop (bar)", predictionCol="prediction", metricName="r2")

# Evaluation of Gradient Boosting
gbt_rmse = evaluator_rmse.evaluate(gbt_predictions)
gbt_r2 = evaluator_r2.evaluate(gbt_predictions)

print(f"Gradient Boosting - Root Mean Squared Error (RMSE): {gbt_rmse}")
print(f"Gradient Boosting - R^2 Score: {gbt_r2}")

# -------------------- Sensitivity Analysis -------------------- #
# For the sensitivity analysis, we will predict both Pressure Drop as well as Energy and Cost
print("\n==> Predictions for Sensitivity Analysis")

# Predict Pressure Drop
gbt_predictions = gbt_model.transform(test_data)

# Add predictions for Energy (kWh) and Cost (€) in the prediction DataFrame
gbt_predictions = gbt_predictions.withColumn('Energy_Predicted (kWh)', F.col('Feed Flowrate (m3/h)') * F.col('prediction') * energy_constant)
gbt_predictions = gbt_predictions.withColumn('Cost_Predicted (€)', F.col('Energy_Predicted (kWh)') * price_per_kwh)

# Show the first predictions for analysis
gbt_predictions.select("Pressure Drop (bar)", "prediction", "Energy_Predicted (kWh)", "Cost_Predicted (€)").show(10)

# -------------------- Analysis of Relationships Between Variables -------------------- #
# View the correlation between key variables: Pressure Drop, kWh, and Cost
print("\n==> Sensitivity Analysis: Correlation between variables")

# Correlation between Pressure Drop and predicted kWh
correlation_pd_kwh = gbt_predictions.stat.corr("Pressure Drop (bar)", "Energy_Predicted (kWh)")
print(f"Correlation between Pressure Drop and Energy (kWh): {correlation_pd_kwh}")

# Correlation between Pressure Drop and predicted Cost
correlation_pd_cost = gbt_predictions.stat.corr("Pressure Drop (bar)", "Cost_Predicted (€)")
print(f"Correlation between Pressure Drop and Cost (€): {correlation_pd_cost}")

# Correlation between predicted kWh and Cost
correlation_kwh_cost = gbt_predictions.stat.corr("Energy_Predicted (kWh)", "Cost_Predicted (€)")
print(f"Correlation between Energy (kWh) and Cost (€): {correlation_kwh_cost}")

# -------------------- Sensitivity Visualization -------------------- #
# Visualize the relationships between Pressure Drop, kWh, and Cost using matplotlib
import matplotlib.pyplot as plt

# Convert to Pandas for visualization with matplotlib
gbt_pd = gbt_predictions.select("Pressure Drop (bar)", "Energy_Predicted (kWh)", "Cost_Predicted (€)").toPandas()

# Plot the relationship between Pressure Drop and Energy (kWh)
plt.figure(figsize=(10,6))
plt.scatter(gbt_pd["Pressure Drop (bar)"], gbt_pd["Energy_Predicted (kWh)"], c='blue', label='Energy (kWh)')
plt.xlabel("Pressure Drop (bar)")
plt.ylabel("Energy (kWh)")
plt.title("Relationship between Pressure Drop and Energy (kWh)")
plt.grid(True)
plt.legend()
plt.show()

# Plot the relationship between Pressure Drop and Cost (€)
plt.figure(figsize=(10,6))
plt.scatter(gbt_pd["Pressure Drop (bar)"], gbt_pd["Cost_Predicted (€)"], c='green', label='Cost (€)')
plt.xlabel("Pressure Drop (bar)")
plt.ylabel("Cost (€)")
plt.title("Relationship between Pressure Drop and Cost (€)")
plt.grid(True)
plt.legend()
plt.show()

