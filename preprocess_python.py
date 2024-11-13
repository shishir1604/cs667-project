# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# Define file paths for AQI and pollutant data
years = [2023,2022]
aqi_files = {
    2023:"/Users/shishir/Downloads/cs667-project/AQI DATA/AQI_daily_city_level_kanpur_2023_kanpur_2023.xlsx",
    2022: "/Users/shishir/Downloads/cs667-project/AQI DATA/AQI_daily_city_level_kanpur_2022_kanpur_2022.xlsx",
   
    
}
pollutant_files = {
    2023:"/Users/shishir/Downloads/cs667-project/POLLUTANT DATA/Raw_data_1Day_2023_site_5500_FTI_Kidwai_Nagar_Kanpur_UPPCB_1Day.csv",
    2022: "/Users/shishir/Downloads/cs667-project/POLLUTANT DATA/Raw_data_1Day_2022_site_5500_FTI_Kidwai_Nagar_Kanpur_UPPCB_1Day.csv",
  
}

# Prepare an empty list to store merged data for each year
merged_yearly_data = []

# Month mapping for reshaping AQI data
month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, 
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}

# Loop through each year to load, process, and merge data
for year in years:
    # Load AQI and pollutant data for the year
    aqi_data = pd.read_excel(aqi_files[year])
    pollutant_data = pd.read_csv(pollutant_files[year])
    
    # Convert the timestamp in pollutant data to datetime and set as index
    pollutant_data['Timestamp'] = pd.to_datetime(pollutant_data['Timestamp'])
    pollutant_data.set_index('Timestamp', inplace=True)
    pollutant_data_interpolated = pollutant_data.interpolate(method='time')
    # Resample pollutant data to daily averages
    daily_pollutant_data = pollutant_data.resample('D').mean().reset_index()
    
    # Reshape AQI data to long format
    aqi_data = aqi_data.rename(columns={"Date": "Day"})
    aqi_long = pd.melt(aqi_data, id_vars=['Day'], var_name='Month', value_name='AQI')
    aqi_long['Month'] = aqi_long['Month'].map(month_mapping)
    aqi_long['Year'] = year
    # Clean the data to remove any non-numeric values
    aqi_long = aqi_long.dropna(subset=['Day', 'Month', 'Year'])  # Drop rows with missing values in date columns
    aqi_long['Day'] = pd.to_numeric(aqi_long['Day'], errors='coerce')
    aqi_long['Month'] = pd.to_numeric(aqi_long['Month'], errors='coerce')
    aqi_long['Year'] = pd.to_numeric(aqi_long['Year'], errors='coerce')
    aqi_long = aqi_long.dropna(subset=['Day', 'Month', 'Year'])  # Drop rows where conversion failed

    # Convert to datetime
    aqi_long['Date'] = pd.to_datetime(aqi_long[['Year', 'Month', 'Day']], errors='coerce')
    aqi_long = aqi_long.dropna(subset=['Date'])  # Drop rows where date conversion failed
    aqi_long.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)

    # Merge daily pollutant data with AQI data
    merged_data = pd.merge(daily_pollutant_data, aqi_long, left_on='Timestamp', right_on='Date')
    merged_data.drop(columns=['Date'], inplace=True)
    # Append merged data to the list
    merged_yearly_data.append(merged_data)

# Concatenate all years into a single DataFrame
combined_data = pd.concat(merged_yearly_data, ignore_index=True)
combined_data=combined_data.fillna(0, inplace=False)

# Display the final merged data structure
# combined_data.to_csv('/Users/shishir/Downloads/cs667-project/combined_data.csv', index=False)
# print(combined_data.head())
columns_to_keep = ['Timestamp', 'PM2.5 (µg/m³)', 'PM10 (µg/m³)','NH3 (µg/m³)','Benzene (µg/m³)','AT (°C)','RH (%)' , 'AQI']

filtered_data = combined_data[columns_to_keep]
print(filtered_data.head())
filtered_data.to_csv('/Users/shishir/Downloads/cs667-project/filtered_data.csv', index=False)

# Perform time-based interpolation to fill NaN values



# %%
from pandas import read_csv
from matplotlib import pyplot
# load dataset
dataset = read_csv('filtered_data.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 4, 5, 6]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
filtered_data = pd.read_csv('/Users/shishir/Downloads/cs667-project/filtered_data.csv')
train_data, test_data = train_test_split(filtered_data, test_size=0.2, random_state=42)

# Save the split datasets to separate CSV files
train_data.to_csv('/Users/shishir/Downloads/cs667-project/train_data.csv', index=False)
test_data.to_csv('/Users/shishir/Downloads/cs667-project/test_data.csv', index=False)

# Display first few rows of each file to verify
print("Training Data:")
print(train_data.head())

print("\nTesting Data:")
print(test_data.head())

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing  import MinMaxScaler
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import LSTM, Dense


# %%
# Load the dataset (assuming 'filtered_combined_data.csv' contains the necessary columns)
data =pd.read_csv('/Users/shishir/Downloads/cs667-project/filtered_data.csv')

# Sort data by timestamp
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data = data.sort_values('Timestamp').set_index('Timestamp')

# Select relevant columns for training
features = ['PM2.5 (µg/m³)', 'PM10 (µg/m³)','NH3 (µg/m³)','Benzene (µg/m³)','AT (°C)','RH (%)' ]
target=['AQI']

# Scale features and target separately
scaler_features = MinMaxScaler()
scaled_features = scaler_features.fit_transform(data[features])

scaler_target = MinMaxScaler()
scaled_target = scaler_target.fit_transform(data[target])

# Convert back to DataFrames for easier manipulation
scaled_features_df = pd.DataFrame(scaled_features, columns=features, index=data.index)
scaled_target_df = pd.DataFrame(scaled_target, columns=target, index=data.index)

# Concatenate scaled features and target into one DataFrame for sequence creation
scaled_data = pd.concat([scaled_features_df, scaled_target_df], axis=1)
print(scaled_data.head())
scaled_data.to_csv('/Users/shishir/Downloads/cs667-project/scaled_data.csv', index=False)
def create_sequences(data, target, lookback=30):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i].values)  # Only scaled features
        y.append(target[i])  # Only target AQI
    return np.array(X), np.array(y)

# Set target as 'AQI' and use only feature columns as predictors
X, y = create_sequences(scaled_features_df, scaled_target_df['AQI'].values)

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# %%
from keras.callbacks import EarlyStopping
from keras.layers import Dropout

# Define the model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))  # Add dropout for regularization
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer for AQI prediction

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with more epochs and early stopping
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, 
                    callbacks=[early_stopping], verbose=1)


# %%
# history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)


# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from numpy import concatenate
# Make predictions
predictions = model.predict(X_test)

# Since we are predicting only 'AQI', use the scaler_target directly for inverse scaling
# Reshape y_test and predictions to be 2D arrays so they can be inverse-transformed
y_test_reshaped = y_test.reshape(-1, 1)
predictions_reshaped = predictions.reshape(-1, 1)

# Reverse scaling for predictions and actual values to get the original AQI scale
y_test_unscaled = scaler_target.inverse_transform(y_test_reshaped).flatten()
predictions_unscaled = scaler_target.inverse_transform(predictions_reshaped).flatten()
print(y_test_unscaled)
print(predictions_unscaled)


# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions_unscaled))
print(f"RMSE: {rmse}")



# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.scatter(y_test_unscaled, predictions_unscaled, alpha=0.5)
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Predicted vs Actual AQI Values')
plt.plot([min(y_test_unscaled), max(y_test_unscaled)], [min(y_test_unscaled), max(y_test_unscaled)], color='red')  # Diagonal line
plt.show()

# Time Series Plot of Predicted vs Actual AQI
plt.figure(figsize=(15, 6))
plt.plot(y_test_unscaled, label='Actual AQI')
plt.plot(predictions_unscaled, label='Predicted AQI')
plt.xlabel('Time')
plt.ylabel('AQI')
plt.title('Time Series of Actual vs Predicted AQI')
plt.legend()
plt.show()


# %%
model.save('/Users/shishir/Downloads/cs667-project/aqi_prediction_model.keras')



