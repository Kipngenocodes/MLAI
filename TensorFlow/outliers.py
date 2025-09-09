# Importing dependencies
import pandas as pd
from scipy.stats import skew

# Adjust display settings
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Load dataset with error handling
try:
    training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display basic statistics
print("Basic Statistics of the Dataset:")
print(training_df.describe())

# Function to detect outliers using IQR
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return len(outliers)

# Analyze potential outliers of the game features
columns_to_check = ['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
print("\nOutlier Analysis:")
for col in columns_to_check:
    num_outliers = detect_outliers(training_df, col)  
    skewness = skew(training_df[col].dropna())
    print(f"{col}: {num_outliers} outliers, skewness = {skewness:.1f}")

print("""
The following columns might contain outliers based on high standard deviation relative to the mean
and a large difference between the 75th percentile and maximum value:

  * total_rooms
  * total_bedrooms
  * population
  * households
  * possibly, median_income (due to moderate skewness)
""")
# more analysis can be done by visualizing the outliers