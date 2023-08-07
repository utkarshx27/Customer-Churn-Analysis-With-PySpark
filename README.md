# Customer Churn Analysis with PySpark

![PySpark Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/1200px-Apache_Spark_logo.svg.png)

This project aims to predict customer churn using PySpark, a powerful and scalable data processing and machine learning library for Apache Spark. Customer churn, also known as customer attrition, is a critical business metric that measures the rate at which customers stop doing business with a company. By analyzing historical customer data, we build a predictive model to identify potential churners and make informed business decisions to retain customers.

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Preparation](#feature-preparation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Deployment](#model-deployment)
- [Recommendations](#recommendations)
- [License](#license)

## Getting Started

To get started with this project, follow the instructions below to set up the required environment and run the code.

## Prerequisites

- Python 3.x
- PySpark
- pandas
- matplotlib
- plotly

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/customer-churn-analysis.git
```

2. Install the required Python libraries:

```
# importing spark session
from pyspark.sql import SparkSession

# data visualization modules
import matplotlib.pyplot as plt
import plotly.express as px

# pandas module
import pandas as pd

# pyspark SQL functions
from pyspark.sql.functions import col, when, count, udf

# pyspark data preprocessing modules
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder

# pyspark data modeling and model evaluation modules
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```

## Data Loading and Preprocessing

In this step, we load the customer churn dataset and perform initial data preprocessing tasks, such as handling missing values and removing outliers.

## Exploratory Data Analysis (EDA)

EDA is performed to gain insights into the dataset's distribution, correlation, and missing values. Data visualization techniques using matplotlib and plotly are used to understand the data better.

## Feature Preparation

Numerical and categorical features are prepared for model training. Numerical features are assembled and scaled, while categorical features are indexed and assembled into a single feature vector.

## Model Training

We use the DecisionTreeClassifier as our predictive model to identify potential churners. The dataset is split into training and test sets for model evaluation.

## Model Evaluation

The model's performance is evaluated using the area under the ROC curve (AUC) metric on both the test and training sets.

## Hyperparameter Tuning

We tune the `maxDepth` hyperparameter of the DecisionTreeClassifier to find the optimal value that results in the best model performance.

## Model Deployment

The trained model can be deployed for making churn predictions on new data.

## Recommendations

Based on the model results, we analyze feature importance and visualize the impact of contract type on churn. Recommendations are provided to reduce customer churn.



