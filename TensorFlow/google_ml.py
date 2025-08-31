# Loading the dependencies...
# Data
import pandas as pd
import numpy as np

# MACHINE LEARNING
import tensorflow as tf
import keras
import ml_edu.experiment
import ml_edu.results

# Data visualization
import plotly.express as px

# Loading the dataset
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

# Updates dataframe to use specific columns.
training_df = chicago_taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.head(200)