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

# Check the first few rows of the split datasets
print(X_train_santa_ynez.head(), X_train_alisal.head())

kernel = (ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-2, 1e4)) * 
          RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) +  
          WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1e3))) 

# Create the GPR pipeline with the kernel and the alpha for numerical stability
gpr_pipeline = make_pipeline(
    StandardScaler(),
    GaussianProcessRegressor(kernel=kernel, alpha=1e-3)  
)


# Fit the model on your training data

gpr_pipeline_santa_ynez = gpr_pipeline.fit(X_train_santa_ynez, y_train_santa_ynez)
santa_ynez_gpr_model_path = os.path.join('res_tool', 'pkl_models', 'gpr_santa_ynez.pkl')
with open(santa_ynez_gpr_model_path, 'wb') as file:
    pickle.dump(gpr_pipeline_santa_ynez, file)

'''# Print hyperparameters after training
trained_kernel = gpr_pipeline.named_steps['gaussianprocessregressor'].kernel_
print("Trained kernel hyperparameters:")
print(trained_kernel)'''

# Evaluate the model on test data
y_pred, std = gpr_pipeline_santa_ynez.predict(X_test_santa_ynez, return_std=True)

# Calculate performance metrics
y_pred_santa_ynez, rmse_santa_ynez, mae_santa_ynez, mape_santa_ynez = evaluate_model(gpr_pipeline_santa_ynez, X_test_santa_ynez, y_test_santa_ynez)

# Output the performance metrics
print("Santa Ynez - RMSE:", rmse_santa_ynez, "MAE:", mae_santa_ynez, "MAPE:", mape_santa_ynez)

# Repeat the process for Alisal data
gpr_pipeline_alisal = gpr_pipeline.fit(X_train_alisal, y_train_alisal)
alisal_gpr_model_path = os.path.join('res_tool', 'pkl_models', 'gpr_alisal.pkl')
with open(alisal_gpr_model_path, 'wb') as file:
    pickle.dump(gpr_pipeline_alisal, file)
# Evaluate the model on Alisal test data and get standard deviation
y_pred_alisal, std_alisal = gpr_pipeline_alisal.predict(X_test_alisal, return_std=True)

# Then calculate the performance metrics
y_pred_alisal, rmse_alisal, mae_alisal, mape_alisal = evaluate_model(gpr_pipeline_alisal, X_test_alisal, y_test_alisal)
# Printing evaluation metrics
# Output the performance metrics
print("alisal - RMSE:", rmse_alisal, "MAE:", mae_alisal, "MAPE:", mape_alisal)





print("Lengths:", len(y_pred_santa_ynez), len(std), len(df_santa_ynez_test['datetime']))

# Assuming y_pred_santa_ynez and std have the same length
if len(y_pred_santa_ynez) == len(std) == len(df_santa_ynez_test['datetime']):
    plt.figure(figsize=(15, 5))
    plt.plot(df_santa_ynez_test['datetime'], y_test_santa_ynez, label='Actual Santa Ynez', alpha=0.7, color='blue')
    plt.plot(df_santa_ynez_test['datetime'], y_pred_santa_ynez, label='Predicted Santa Ynez (GP)', alpha=0.7, color='cyan')
    plt.fill_between(df_santa_ynez_test['datetime'], y_pred_santa_ynez - 1.96*std, y_pred_santa_ynez + 1.96*std, color='cyan', alpha=0.2)
    plt.title('Gaussian Process Model: Actual vs Predicted Reservoir Water Levels (Santa Ynez)')
    plt.xlabel('Date')
    plt.ylabel('Reservoir Water Level')
    plt.legend()
    plt.show()
else:
    print("Mismatch in array lengths. Cannot plot confidence intervals.")


# Now you use alisal for plotting the confidence intervals
plt.figure(figsize=(15, 5))
plt.plot(df_alisal_test['datetime'], y_test_alisal, label='Actual Alisal', alpha=0.7, color='red')
plt.plot(df_alisal_test['datetime'], y_pred_alisal, label='Predicted Alisal (GP)', alpha=0.7, color='orange')
plt.fill_between(df_alisal_test['datetime'], y_pred_alisal - 1.96*std_alisal, y_pred_alisal + 1.96*std_alisal, color='orange', alpha=0.2)
plt.title('Gaussian Process Model: Actual vs Predicted Reservoir Water Levels (Alisal)')
plt.xlabel('Date')
plt.ylabel('Reservoir Water Level')
plt.legend()
plt.show()