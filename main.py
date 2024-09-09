# Import the necessary libraries

# For Data loading, Exploratory Data Analysis, Graphing
import pandas as pd   # Pandas for data processing libraries
import numpy as np    # Numpy for mathematical functions

import matplotlib.pyplot as plt # Matplotlib for visualization tasks
import seaborn as sns # Seaborn for data visualization library based on matplotlib.

import sklearn        # ML tasks
from sklearn.model_selection import train_test_split # Split the dataset
from sklearn.metrics import mean_squared_error  # Calculate Mean Squared Error

# Build the Network
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# Load the CSV file with specified encoding to avoid errors
url = '/Users/vinayak/Documents/GitHub/M-fine/M-fine/sales_data_sample.csv'
advertising_df = pd.read_csv(url, index_col=0, encoding='latin1')

# Get concise summary and shape of the dataframe
advertising_df.info()
print(advertising_df.describe())
print("Shape of the dataset:", advertising_df.shape)
print(advertising_df.head())

# Check for missing values
missing_values = advertising_df.isnull().sum()
print("Missing values:\n", missing_values)

# Drop rows with missing values
advertising_df.dropna(inplace=True)

# Convert date columns to datetime, if applicable (replace 'date_column' with the actual column name)
# advertising_df['date_column'] = pd.to_datetime(advertising_df['date_column'], errors='coerce')

# Drop non-numeric columns before computing correlation matrix (assuming 'date_column' and similar columns)
non_numeric_columns = advertising_df.select_dtypes(exclude=[np.number]).columns
advertising_df_numeric = advertising_df.drop(columns=non_numeric_columns)

## Exploratory Data Analysis (EDA)

# Plot a heatmap for the correlation matrix
plt.figure(figsize=(10, 5))
sns.heatmap(advertising_df_numeric.corr(), annot=True, vmin=-1, vmax=1, cmap='ocean')
plt.title("Correlation Heatmap")
plt.show()

# Create a correlation matrix for values >= 0.5 or <= -0.7
corr = advertising_df_numeric.corr()
plt.figure(figsize=(10, 5))
sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.7)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, 
            linewidths=0.1, annot=True, annot_kws={"size": 8}, square=True)
plt.title("Filtered Correlation Matrix (Threshold: 0.5 and -0.7)")
plt.tight_layout()
plt.show()
advertising_df.corr()

### Visualize Correlation

# Generate a mask for the upper triangle
mask = np.zeros_like(advertising_df.corr(), dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(advertising_df.corr(), mask=mask, cmap=cmap, vmax=.9, square=True, linewidths=.5, ax=ax)

# Display the correlation matrix
plt.title("Correlation Matrix")
plt.show()

'''=== Show the linear relationship between features  and sales Thus, it provides that how the scattered
      they are and which features has more impact in prediction of house price. ==='''

# visiualize all variables  with sales
from scipy import stats
#creates figure
plt.figure(figsize=(18, 18))

for i, col in enumerate(advertising_df.columns[0:13]): #iterates over all columns except for price column (last one)
    plt.subplot(5, 3, i+1) # each row three figure
    x = advertising_df[col] #x-axis
    y = advertising_df['sales'] #y-axis
    plt.plot(x, y, 'o')

    # Create regression line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1)) (np.unique(x)), color='red')
    plt.xlabel(col) # x-label
    plt.ylabel('sales') # y-label   
    plt.title(f'{col} vs sales') # title
    plt.tight_layout()
    plt.show()


from sklearn.preprocessing import LabelEncoder

# Encoding categorical columns to numeric
label_encoder = LabelEncoder()

# Apply LabelEncoder on all categorical columns
for col in ['CITY', 'COUNTRY', 'PRODUCTLINE', 'CUSTOMERNAME', 'DEALSIZE']:
    advertising_df[col] = label_encoder.fit_transform(advertising_df[col])

# Handle the date column (assuming 'date_column' exists)
advertising_df['ORDERDATE'] = pd.to_datetime(advertising_df['ORDERDATE'], errors='coerce')

# You can extract year, month, day for the date
advertising_df['YEAR'] = advertising_df['ORDERDATE'].dt.year
advertising_df['MONTH'] = advertising_df['ORDERDATE'].dt.month
advertising_df['DAY'] = advertising_df['ORDERDATE'].dt.day

# Drop the original date column if not needed anymore
advertising_df.drop(columns=['ORDERDATE'], inplace=True)

# Now use the processed data for model training
X = advertising_df[['PRICE', 'CITY', 'COUNTRY', 'PRODUCTLINE', 'CUSTOMERNAME', 'DEALSIZE', 'YEAR', 'MONTH', 'DAY']]
y = advertising_df['SALES']

# Normalize the features
normalized_feature = keras.utils.normalize(X.values)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(normalized_feature, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Build the ANN model
model = Sequential()
model.add(Dense(8, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

# Compile Model
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Fit the Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=32)

# Plot model loss
plt.figure(figsize=(15, 8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss (MSE) on Training and Validation Data')
plt.ylabel('Loss-Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
plt.show()