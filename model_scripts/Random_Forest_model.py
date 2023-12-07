from model_creation_scripts.database_connection import connect_database
from model_creation_scripts.data_fetching import fetch_data
from model_creation_scripts.feature_engineering import create_lag_features
from model_creation_scripts.modeling_utilities import *
from model_creation_scripts.data_preparation import prepare_dataset

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return y_pred, rmse, mae, mape

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

# Initialize the RandomForestRegressor model with default parameters
rf_santa_ynez = RandomForestRegressor(random_state=0)

# Train the model for Santa Ynez reservoir
rf_santa_ynez.fit(X_train_santa_ynez, y_train_santa_ynez)
santa_ynez_rf_model_path = os.path.join('res_tool', 'pkl_models', 'rf_santa_ynez.pkl')
with open(santa_ynez_rf_model_path, 'wb') as file:
    pickle.dump(rf_santa_ynez, file)
y_pred_santa_ynez, rmse_santa_ynez, mae_santa_ynez, mape_santa_ynez = evaluate_model(
    rf_santa_ynez, X_test_santa_ynez, y_test_santa_ynez)

print(f"Santa Ynez Reservoir - RMSE: {rmse_santa_ynez}, MAE: {mae_santa_ynez}, MAPE: {mape_santa_ynez}")

# Train the model for Alisal reservoir
rf_alisal = RandomForestRegressor(
    max_depth=44, 
    min_samples_leaf=1, 
    min_samples_split=2, 
    n_estimators=153, 
    random_state=0
)

# Evaluate the model on the test set for Alisal
rf_alisal.fit(X_train_alisal, y_train_alisal)
alisal_rf_model_path = os.path.join('res_tool', 'pkl_models', 'rf_alisal.pkl')
with open(alisal_rf_model_path, 'wb') as file:
    pickle.dump(rf_alisal, file)

y_pred_alisal, rmse_alisal, mae_alisal, mape_alisal = evaluate_model(
    rf_alisal, X_test_alisal, y_test_alisal)
print(f"Alisal Reservoir - RMSE: {rmse_alisal}, MAE: {mae_alisal}, MAPE: {mape_alisal}")


# Plotting the actual vs predicted values for Santa Ynez using RandomForest
plt.figure(figsize=(15, 5))
plt.plot(df_santa_ynez_test['datetime'], y_test_santa_ynez, label='Actual Santa Ynez', alpha=0.7, color='blue')
plt.plot(df_santa_ynez_test['datetime'], y_pred_santa_ynez, label='Predicted Santa Ynez (RandomForest)', alpha=0.7, color='green')
plt.title('RandomForest Model: Actual vs Predicted Reservoir Water Levels (Santa Ynez)')
plt.xlabel('Date')
plt.ylabel('Reservoir Water Level')
plt.legend()
plt.show()

# Plotting the actual vs predicted values for Alisal using RandomForest
plt.figure(figsize=(15, 5))
plt.plot(df_alisal_test['datetime'], y_test_alisal, label='Actual Alisal', alpha=0.7, color='red')
plt.plot(df_alisal_test['datetime'], y_pred_alisal, label='Predicted Alisal (RandomForest)', alpha=0.7, color='purple')
plt.title('RandomForest Model: Actual vs Predicted Reservoir Water Levels (Alisal)')
plt.xlabel('Date')
plt.ylabel('Reservoir Water Level')
plt.legend()
plt.show()




# Assuming the param_grid is already defined as shown earlier
'''param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10)
}

# Randomized search for Santa Ynez
random_search_santa_ynez = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=0),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    verbose=1,
    random_state=0,
    n_jobs=-1
)

# Train the model for Santa Ynez
random_search_santa_ynez.fit(X_train_santa_ynez, y_train_santa_ynez)
# Randomized search for Alisal
random_search_alisal = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=0),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    verbose=1,
    random_state=0,
    n_jobs=-1
)

# Train the model for Alisal
random_search_alisal.fit(X_train_alisal, y_train_alisal)


best_params_santa_ynez_random = random_search_santa_ynez.best_params_
best_params_alisal_random = random_search_alisal.best_params_

print( best_params_santa_ynez_random, best_params_alisal_random)  '''  # Print the best parameters 
# Grid search for Santa Ynez
'''grid_search_santa_ynez = GridSearchCV(
    estimator=RandomForestRegressor(random_state=0),
    param_grid=param_grid,
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Train the model for Santa Ynez
grid_search_santa_ynez.fit(X_train_santa_ynez, y_train_santa_ynez)

# Grid search for Alisal
grid_search_alisal = GridSearchCV(
    estimator=RandomForestRegressor(random_state=0),
    param_grid=param_grid,
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Train the model for Alisal
grid_search_alisal.fit(X_train_alisal, y_train_alisal)

best_params_santa_ynez_grid = grid_search_santa_ynez.best_params_

best_params_alisal_grid = grid_search_alisal.best_params_

print( best_params_santa_ynez_grid, best_params_alisal_grid)    # Print the best parameters'''