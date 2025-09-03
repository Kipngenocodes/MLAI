# Training  a binary classifier, calculating metrics for binary classifier at different thresholds
# importing and loading dependencies
import keras 
import ml_edu.experiment
import ml_edu.results
import pandas as pd
import numpy as np
import plotly.express as px

# adjusting the granularity of reporting
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# import dataset
rice_dataset_raw = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")

# viewing the dataset
print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(rice_dataset_raw.index)))


# Read and provide statistics on the dataset.
rice_dataset = rice_dataset_raw[[
        'Area',
        'Perimeter',
        'Major_Axis_Length',
        'Minor_Axis_Length',
        'Eccentricity',
        'Convex_Area',
        'Extent',
        'Class',
]]

rice_dataset.describe()

# Data description

print(
    f'The shortest grain is {rice_dataset.Major_Axis_Length.min():.1f}px long,'
    f' while the longest is {rice_dataset.Major_Axis_Length.max():.1f}px.'
)
print(
    f'The smallest rice grain has an area of {rice_dataset.Area.min()}px, while'
    f' the largest has an area of {rice_dataset.Area.max()}px.'
)
print(
    'The largest rice grain, with a perimeter of'
    f' {rice_dataset.Perimeter.max():.1f}px, is'
    f' ~{(rice_dataset.Perimeter.max() - rice_dataset.Perimeter.mean())/rice_dataset.Perimeter.std():.1f} standard'
    f' deviations ({rice_dataset.Perimeter.std():.1f}) from the mean'
    f' ({rice_dataset.Perimeter.mean():.1f}px).'
)
print(
    f'This is calculated as: ({rice_dataset.Perimeter.max():.1f} -'
    f' {rice_dataset.Perimeter.mean():.1f})/{rice_dataset.Perimeter.std():.1f} ='
    f' {(rice_dataset.Perimeter.max() - rice_dataset.Perimeter.mean())/rice_dataset.Perimeter.std():.1f}'
)

# Exploring the dataset