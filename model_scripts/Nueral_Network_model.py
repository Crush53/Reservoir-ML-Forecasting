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

mlp_regressor_santa_ynez = MLPRegressor(
    hidden_layer_sizes=(140, 90), 
    activation='relu', 
    solver='adam', 
    alpha=0.0001, 
    batch_size='auto', 
    learning_rate='constant', 
    learning_rate_init=0.001, 
    power_t=0.5, 
    max_iter=250, 
    shuffle=True, 
    random_state=None, 
    tol=0.0001, 
    verbose=False, 
    warm_start=False, 
    momentum=0.9, 
    nesterovs_momentum=True, 
    early_stopping=False, 
    validation_fraction=0.1, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=1e-08, 
    n_iter_no_change=10, 
    max_fun=15000
)

# For Santa Ynez Reservoir
mlp_regressor_santa_ynez.fit(X_train_santa_ynez, y_train_santa_ynez)
santa_ynez_mlp_model_path = os.path.join('res_tool', 'pkl_models', 'mlp_santa_ynez.pkl')
with open(santa_ynez_mlp_model_path, 'wb') as file:
    pickle.dump(mlp_regressor_santa_ynez, file)


y_pred_santa_ynez, rmse_santa_ynez, mae_santa_ynez, mape_santa_ynez = evaluate_model(mlp_regressor_santa_ynez, X_test_santa_ynez, y_test_santa_ynez)

print(f"Santa Ynez Reservoir - RMSE: {rmse_santa_ynez}, MAE: {mae_santa_ynez}, MAPE: {mape_santa_ynez}")





# For Alisal Reservoir

mlp_regressor_alisal = MLPRegressor(
    activation='identity',
    alpha=0.0003962735057040824,
    hidden_layer_sizes=(150, 100),
    learning_rate='constant',
    max_iter=770,
    solver='lbfgs',
    random_state=42  # Adding a random state for reproducibility
)

mlp_regressor_alisal.fit(X_train_alisal, y_train_alisal)
alisal_mlp_model_path = os.path.join('res_tool', 'pkl_models', 'mlp_alisal.pkl')
# Serialize and save the Alisal MLP Regressor model
with open(alisal_mlp_model_path, 'wb') as file:
    pickle.dump(mlp_regressor_alisal, file)

y_pred_alisal, rmse_alisal, mae_alisal, mape_alisal = evaluate_model(mlp_regressor_alisal, X_test_alisal, y_test_alisal)

print(f"Alisal Reservoir - RMSE: {rmse_alisal}, MAE: {mae_alisal}, MAPE: {mape_alisal}")





'''#random SEARCH to check for best params
param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (150, 100)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': uniform(0.0001, 0.001),
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter': randint(250, 1000)
}

# Setup RandomizedSearchCV for Santa Ynez
mlp_reg_santa_ynez = MLPRegressor()
random_search_santa_ynez = RandomizedSearchCV(mlp_reg_santa_ynez, param_dist, n_iter=100, cv=5, verbose=1, n_jobs=-1, random_state=42)

# Setup RandomizedSearchCV for Alisal
mlp_reg_alisal = MLPRegressor()
random_search_alisal = RandomizedSearchCV(mlp_reg_alisal, param_dist, n_iter=100, cv=5, verbose=1, n_jobs=-1, random_state=42)

# Train for Santa Ynez
random_search_santa_ynez.fit(X_train_santa_ynez, y_train_santa_ynez)

# Train for Alisal
random_search_alisal.fit(X_train_alisal, y_train_alisal)

# Get the best parameters
best_model_santa_ynez = random_search_santa_ynez.best_estimator_
best_model_alisal = random_search_alisal.best_estimator_

# Best parameters and scores for Santa Ynez
best_params_santa_ynez = random_search_santa_ynez.best_params_
best_score_santa_ynez = random_search_santa_ynez.best_score_
print("Best Parameters for Santa Ynez Reservoir:", best_params_santa_ynez)
print("Best Score for Santa Ynez Reservoir:", best_score_santa_ynez)

# Best parameters and scores for Alisal
best_params_alisal = random_search_alisal.best_params_
best_score_alisal = random_search_alisal.best_score_
print("Best Parameters for Alisal Reservoir:", best_params_alisal)
print("Best Score for Alisal Reservoir:", best_score_alisal)'''


plt.figure(figsize=(15, 5))
plt.plot(df_santa_ynez_test['datetime'], y_test_santa_ynez, label='Actual Santa Ynez', alpha=0.7, color='blue')
plt.plot(df_santa_ynez_test['datetime'], y_pred_santa_ynez, label='Predicted Santa Ynez (MLPRegressor)', alpha=0.7, color='green')
plt.title('MLPRegressor Model: Actual vs Predicted Reservoir Water Levels (Santa Ynez)')
plt.xlabel('Date')
plt.ylabel('Reservoir Water Level')
plt.legend()
plt.show()

# Plotting the actual vs predicted values for Alisal using MLPRegressor
plt.figure(figsize=(15, 5))
plt.plot(df_alisal_test['datetime'], y_test_alisal, label='Actual Alisal', alpha=0.7, color='red')
plt.plot(df_alisal_test['datetime'], y_pred_alisal, label='Predicted Alisal (MLPRegressor)', alpha=0.7, color='purple')
plt.title('MLPRegressor Model: Actual vs Predicted Reservoir Water Levels (Alisal)')
plt.xlabel('Date')
plt.ylabel('Reservoir Water Level')
plt.legend()
plt.show()