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

dt_pipeline = make_pipeline(
    StandardScaler(),
    DecisionTreeRegressor(random_state=0)
)
#Parameter tuning for the model
'''param_grid_alisal = {
    'decisiontreeregressor__max_depth': [18, 19, 20, 21, 22],
    'decisiontreeregressor__min_samples_split': [2, 3, 4, 5],
    'decisiontreeregressor__min_samples_leaf': [1, 2, 3, 4],
    # Add any other parameters you wish to tune
}

grid_search_alisal = GridSearchCV(
    estimator=dt_pipeline,
    param_grid=param_grid_alisal,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

# Fit the GridSearchCV (this will take some time)
grid_search_alisal.fit(X_train_alisal, y_train_alisal)  # Use your training data here

# Get the best parameters
best_parameters = grid_search_alisal.best_params_

# Evaluate the best model from grid search
y_pred, rmse, mae, mape = evaluate_model(grid_search_alisal.best_estimator_, X_test_alisal, y_test_alisal)

print(best_parameters, rmse, mae, mape)'''


# Initialize the Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(
    max_depth=20,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=0
)

# Train the model for Santa Ynez reservoir
dt_regressor.fit(X_train_santa_ynez, y_train_santa_ynez)
santa_ynez_dt_model_path = os.path.join('res_tool', 'pkl_models', 'dt_santa_ynez.pkl')
with open(santa_ynez_dt_model_path, 'wb') as file:
    pickle.dump(dt_regressor, file)
# Evaluate the model on the test set for Santa Ynez
y_pred_santa_ynez, rmse_santa_ynez, mae_santa_ynez, mape_santa_ynez = evaluate_model(
    dt_regressor, X_test_santa_ynez, y_test_santa_ynez)


dt_regressor_alisal = DecisionTreeRegressor(
    max_depth=19,
    min_samples_leaf=1,
    min_samples_split=4,
    random_state=0
)
# Train the model for Alisal reservoir
dt_regressor_alisal.fit(X_train_alisal, y_train_alisal)
alisal_dt_model_path = os.path.join('res_tool', 'pkl_models', 'dt_alisal.pkl')
with open(alisal_dt_model_path, 'wb') as file:
    pickle.dump(dt_regressor_alisal, file)
# Evaluate the model on the test set for Alisal
y_pred_alisal, rmse_alisal, mae_alisal, mape_alisal = evaluate_model(
    dt_regressor_alisal, X_test_alisal, y_test_alisal)

# Gather the evaluation metrics
eval_metrics_santa_ynez = {
    'RMSE-santa ynez': rmse_santa_ynez,
    'MAE-santa ynez': mae_santa_ynez,
    'MAPE-santa ynez': mape_santa_ynez
}

eval_metrics_alisal = {
    'RMSE-alisal': rmse_alisal,
    'MAE-alisal': mae_alisal,
    'MAPE-alisal': mape_alisal
}

print(eval_metrics_santa_ynez, eval_metrics_alisal)

# Plotting the actual vs predicted values for Santa Ynez using the Decision Tree model
plt.figure(figsize=(15, 5))
plt.plot(df_santa_ynez_test['datetime'], y_test_santa_ynez, label='Actual Santa Ynez', alpha=0.7, color='blue')
plt.plot(df_santa_ynez_test['datetime'], y_pred_santa_ynez, label='Predicted Santa Ynez (DT)', alpha=0.7, color='cyan')
plt.title('Decision Tree Model: Actual vs Predicted Reservoir Water Levels (Santa Ynez)')
plt.xlabel('Date')
plt.ylabel('Reservoir Water Level')
plt.legend()
plt.show()

# Plotting the actual vs predicted values for Alisal using the Decision Tree model
plt.figure(figsize=(15, 5))
plt.plot(df_alisal_test['datetime'], y_test_alisal, label='Actual Alisal', alpha=0.7, color='red')
plt.plot(df_alisal_test['datetime'], y_pred_alisal, label='Predicted Alisal (DT)', alpha=0.7, color='orange')
plt.title('Decision Tree Model: Actual vs Predicted Reservoir Water Levels (Alisal)')
plt.xlabel('Date')
plt.ylabel('Reservoir Water Level')
plt.legend()
plt.show()