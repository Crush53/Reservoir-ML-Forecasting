from model_creation_scripts.database_connection import connect_database
from model_creation_scripts.data_fetching import fetch_data
from model_creation_scripts.feature_engineering import create_lag_features
from model_creation_scripts.modeling_utilities import *  # Import all the machine learning and plotting utilities
from model_creation_scripts.data_preparation import prepare_dataset

# Function to create a pipeline and fit SVR model
def fit_svr(X_train, y_train, C=1000, epsilon=0.2, gamma=0.01):
    regr_pipeline = make_pipeline(StandardScaler(), SVR(C=C, epsilon=epsilon, gamma=gamma))
    regr_pipeline.fit(X_train, y_train)
    return regr_pipeline
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

# Splitting the data for Santa Ynez Reservoir
df_santa_ynez_train = df_combined.loc[(df_combined['stn_id'] == '11122000') & (df_combined['datetime'] <= training_end_date)]
df_santa_ynez_test = df_combined.loc[(df_combined['stn_id'] == '11122000') & (df_combined['datetime'] >= test_start_date)]
X_train_santa_ynez = df_santa_ynez_train[predictor_columns]
y_train_santa_ynez = df_santa_ynez_train['storage_value']
X_test_santa_ynez = df_santa_ynez_test[predictor_columns]
y_test_santa_ynez = df_santa_ynez_test['storage_value']


# Splitting the data for Alisal Reservoir
df_alisal_train = df_combined.loc[(df_combined['stn_id'] == '11128300') & (df_combined['datetime'] <= training_end_date)]
df_alisal_test = df_combined.loc[(df_combined['stn_id'] == '11128300') & (df_combined['datetime'] >= test_start_date)]
X_train_alisal = df_alisal_train[predictor_columns]
y_train_alisal = df_alisal_train['storage_value']
X_test_alisal = df_alisal_test[predictor_columns]
y_test_alisal = df_alisal_test['storage_value']


# Fit and evaluate SVR model for Santa Ynez
svr_pipeline_santa_ynez = fit_svr(X_train_santa_ynez, y_train_santa_ynez, C=1000, epsilon=.85, gamma=0.01)
santa_ynez_model_path = os.path.join('res_tool', 'pkl_models', 'svr_santa_ynez.pkl')
# Serialize and save the Santa Ynez model
with open(santa_ynez_model_path, 'wb') as file:
    pickle.dump(svr_pipeline_santa_ynez, file)
y_pred_santa_ynez, rmse_santa_ynez, mae_santa_ynez, mape_santa_ynez = evaluate_model(svr_pipeline_santa_ynez, X_test_santa_ynez, y_test_santa_ynez)

# Print Santa Ynez evaluation metrics
print("SVR-Santa Ynez - RMSE:", rmse_santa_ynez)
print("SVR-Santa Ynez - MAE:", mae_santa_ynez)
print("SVR-Santa Ynez - MAPE:", mape_santa_ynez)


svr_pipeline_alisal = fit_svr(X_train_alisal, y_train_alisal, C=1000, epsilon=0.1, gamma=0.01)
alisal_model_path = os.path.join('res_tool', 'pkl_models', 'svr_alisal.pkl')
# Serialize and save the Alisal model
with open(alisal_model_path, 'wb') as file:
    pickle.dump(svr_pipeline_alisal, file)

y_pred_alisal, rmse_alisal, mae_alisal, mape_alisal = evaluate_model(svr_pipeline_alisal, X_test_alisal, y_test_alisal)

# Print Alisal evaluation metrics
print("SVR-Alisal - RMSE:", rmse_alisal)
print("SVR-Alisal - MAE:", mae_alisal)
print("SVR-Alisal - MAPE:", mape_alisal)

# Plotting the actual vs predicted values for Santa Ynez using SVR
plt.figure(figsize=(15, 5))
plt.plot(df_santa_ynez_test['datetime'], y_test_santa_ynez, label='Actual Santa Ynez', alpha=0.7, color='blue')
plt.plot(df_santa_ynez_test['datetime'], y_pred_santa_ynez, label='Predicted Santa Ynez (SVR)', alpha=0.7, color='cyan')
plt.title('SVR Model: Actual vs Predicted Reservoir Water Levels (Santa Ynez)')
plt.xlabel('Date')
plt.ylabel('Reservoir Water Level')
plt.legend()
plt.show()

# Plotting the actual vs predicted values for Alisal using SVR
plt.figure(figsize=(15, 5))
plt.plot(df_alisal_test['datetime'], y_test_alisal, label='Actual Alisal', alpha=0.7, color='red')
plt.plot(df_alisal_test['datetime'], y_pred_alisal, label='Predicted Alisal (SVR)', alpha=0.7, color='orange')
plt.title('SVR Model: Actual vs Predicted Reservoir Water Levels (Alisal)')
plt.xlabel('Date')
plt.ylabel('Reservoir Water Level')
plt.legend()
plt.show()

'''
# Define the parameter grid for SVR
param_grid = {
    'svr__C': [1, 10, 100, 1000],
    'svr__epsilon': [0.1, 0.2, 0.5, 1],
    'svr__gamma': ['scale', 'auto', 0.01, 0.001]
}

# Fit GridSearchCV to the training data for Alisal
grid_search_alisal = GridSearchCV(
    make_pipeline(StandardScaler(), SVR()),
    param_grid,
    cv=5,  # Number of cross-validation folds
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1  # Use all available cores
)

grid_search_alisal.fit(X_train_alisal, y_train_alisal)

# Get the best parameters and the corresponding best SVR model for Alisal
best_params_alisal = grid_search_alisal.best_params_
best_svr_model_alisal = grid_search_alisal.best_estimator_

print("Best Parameters for SVR (Alisal):", best_params_alisal)

# Evaluate the best model for Alisal
y_pred_alisal, rmse_alisal, mae_alisal, mape_alisal = evaluate_model(best_svr_model_alisal, X_test_alisal, y_test_alisal)

# Print the evaluation metrics for Alisal
print("SVR-Alisal - Best Model - RMSE:", rmse_alisal)
print("SVR-Alisal - Best Model - MAE:", mae_alisal)
print("SVR-Alisal - Best Model - MAPE:", mape_alisal)
'''


'''#Now trying SVR model

# Initialize the SVR model (you can adjust the parameters as needed)
svr_regressor = SVR(kernel='rbf', C=100, gamma=0.01, epsilon=.5)

# Train the model for Santa Ynez
svr_regressor.fit(X_train_santa_ynez, y_train_santa_ynez)
# Predict on the test set for Santa Ynez
y_pred_santa_ynez_svr = svr_regressor.predict(X_test_santa_ynez)

# Train the model for Alisal
svr_regressor.fit(X_train_alisal, y_train_alisal)
# Predict on the test set for Alisal
y_pred_alisal_svr = svr_regressor.predict(X_test_alisal)
'''
