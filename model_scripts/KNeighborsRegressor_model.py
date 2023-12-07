from model_creation_scripts.database_connection import connect_database
from model_creation_scripts.data_fetching import fetch_data
from model_creation_scripts.feature_engineering import create_lag_features
from model_creation_scripts.modeling_utilities import *
from model_creation_scripts.data_preparation import prepare_dataset




# SQL queries to fetch reservoir and weather data
query_reservoir = "SELECT stn_id, datetime, storage_value FROM res_data;"
query_weather = "SELECT stn_id, datetime, average_temperature, precipitation FROM weather_data;"

# Connect to the database
conn = connect_database()

if conn:
    # Fetch the data
    df_reservoir = fetch_data(query_reservoir, conn)
    df_weather = fetch_data(query_weather, conn)

    # Close the database connection
    conn.close()
else:
    df_reservoir = pd.DataFrame()
    df_weather = pd.DataFrame()

# Define cutoff dates and lag days
training_end_date = '2022-12-31'
test_start_date = '2023-01-01'
lag_days = 14

# Prepare the dataset
df_combined, predictor_columns = prepare_dataset(df_reservoir, df_weather, lag_days)

# Splitting the data for Santa Ynez and Alisal Reservoirs
df_santa_ynez_train = df_combined.loc[(df_combined['stn_id'] == '11122000') & (df_combined['datetime'] <= training_end_date)]
df_santa_ynez_test = df_combined.loc[(df_combined['stn_id'] == '11122000') & (df_combined['datetime'] >= test_start_date)]
X_train_santa_ynez = df_santa_ynez_train[predictor_columns]
y_train_santa_ynez = df_santa_ynez_train['storage_value']
X_test_santa_ynez = df_santa_ynez_test[predictor_columns]
y_test_santa_ynez = df_santa_ynez_test['storage_value']

df_alisal_train = df_combined.loc[(df_combined['stn_id'] == '11128300') & (df_combined['datetime'] <= training_end_date)]
df_alisal_test = df_combined.loc[(df_combined['stn_id'] == '11128300') & (df_combined['datetime'] >= test_start_date)]
X_train_alisal = df_alisal_train[predictor_columns]
y_train_alisal = df_alisal_train['storage_value']
X_test_alisal = df_alisal_test[predictor_columns]
y_test_alisal = df_alisal_test['storage_value']

# Initialize the KNeighborsRegressor model
knn_regressor_ynez = KNeighborsRegressor(n_neighbors=3)  # n_neighbors is a hyperparameter that you may tune
knn_regressor_alisal = KNeighborsRegressor(n_neighbors=3) 
# Train the model
knn_regressor_ynez.fit(X_train_santa_ynez, y_train_santa_ynez)
ynez_knn_model_path = os.path.join('res_tool', 'pkl_models', 'knn_ynez.pkl')
with open(ynez_knn_model_path, 'wb') as file:
    pickle.dump(knn_regressor_ynez, file)

# Predict on the test set
y_pred_santa_ynez = knn_regressor_ynez.predict(X_test_santa_ynez)

# Train the model for Alisal
knn_regressor_alisal.fit(X_train_alisal, y_train_alisal)
alisal_knn_model_path = os.path.join('res_tool', 'pkl_models', 'knn_alisal.pkl')
with open(alisal_knn_model_path, 'wb') as file:
    pickle.dump(knn_regressor_alisal, file)
# Predict on the test set for Alisal
y_pred_alisal = knn_regressor_alisal.predict(X_test_alisal)

# Evaluate the model using RMSE
rmse = sqrt(mean_squared_error(y_test_santa_ynez, y_pred_santa_ynez))
# Calculate MAE
mae = mean_absolute_error(y_test_santa_ynez, y_pred_santa_ynez)
# Calculate MAPE
mape = mean_absolute_percentage_error(y_test_santa_ynez, y_pred_santa_ynez)

#evaluate the model for alisal
rmse_alisal = sqrt(mean_squared_error(y_test_alisal, y_pred_alisal))
mae_alisal = mean_absolute_error(y_test_alisal, y_pred_alisal)
mape_alisal = mean_absolute_percentage_error(y_test_alisal, y_pred_alisal)

# Print metrics
print("Root Mean Squared Error-Ynez:", rmse)
print("Mean Absolute Error-Ynez:", mae)
print("Mean Absolute Percentage Error-Ynez:", mape)
print("Alisal - Root Mean Squared Error:", rmse_alisal)
print("Alisal - Mean Absolute Error:", mae_alisal)
print("Alisal - Mean Absolute Percentage Error:", mape_alisal)


# Plotting the actual vs predicted values for Santa Ynez using the KN model
plt.figure(figsize=(15, 5))
plt.plot(df_santa_ynez_test['datetime'], y_test_santa_ynez, label='Actual Santa Ynez', alpha=0.7, color='blue')
plt.plot(df_santa_ynez_test['datetime'], y_pred_santa_ynez, label='Predicted Santa Ynez (KN)', alpha=0.7, color='cyan')
plt.title('KN Model: Actual vs Predicted Reservoir Water Levels (Santa Ynez)')
plt.xlabel('Date')
plt.ylabel('Reservoir Water Level')
plt.legend()
plt.show()

# Plotting the actual vs predicted values for Alisal using the KN model
plt.figure(figsize=(15, 5))
plt.plot(df_alisal_test['datetime'], y_test_alisal, label='Actual Alisal', alpha=0.7, color='red')
plt.plot(df_alisal_test['datetime'], y_pred_alisal, label='Predicted Alisal (KN)', alpha=0.7, color='orange')
plt.title('KN Model: Actual vs Predicted Reservoir Water Levels (Alisal)')
plt.xlabel('Date')
plt.ylabel('Reservoir Water Level')
plt.legend()
plt.show()

