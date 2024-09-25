# Desalination Plant Machine Learning Models Benchmark with Apache Spark ML in DataBricks
This repository contains a comparison of three machine learning models (Linear Regression, Random Forest, and Gradient Boosting) applied to simulated desalination plant data. The goal is to predict the pressure drop in the reverse osmosis process based on various operational parameters using apache spark in Databricks.

## Project Overview
The need for clean water is increasing, and desalination plays a key role in addressing water scarcity. One of the challenges in reverse osmosis desalination is the pressure drop, which is influenced by physical-chemical and mechanical factors. This repository explores how machine learning models can help optimize the desalination process, particularly by predicting pressure drop more accurately.

The repository includes two main notebooks:

- Desalination_ModelsBenchmark: A comparison of Linear Regression, Random Forest, and Gradient Boosting models.
- Desalination_XGBoost: A detailed exploration of the XGBoost model, including sensitivity analysis and energy/cost calculations.
  
## Dataset
A mock dataset was generated based on insights from the following research articles:

- [Low energy consumption in the Perth seawater desalination plant](https://www.researchgate.net/publication/228491362_Low_energy_consumption_in_the_Perth_seawater_desalination_plant)
- [Membranes for SWRO Desalination](https://www.mdpi.com/2077-0375/11/10/774)
This dataset simulates operational conditions for a desalination plant, including factors like flow rates, salinity, temperature, and energy consumption.

## Models Compared
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor
The notebooks showcase the training times, RMSE (Root Mean Squared Error), and R² scores for each model, providing insights into their performance on this specific dataset.

## Results Summary
- XGBoost had the best accuracy (lowest RMSE, highest R²), though it required more time for training.
- Random Forest and Linear Regression performed reasonably well but were outclassed by XGBoost in prediction accuracy.

## How to Run
1. Clone the repository:

'''
git clone https://github.com/your-username/desalination-ml-benchmark.git
cd desalination-ml-benchmark '''

2. Upload the notebooks to Databricks or any PySpark-enabled environment.

3. Ensure the mock dataset (provided in the repository) is available in the correct path.

4. Run the notebooks to benchmark the models and explore the results.
