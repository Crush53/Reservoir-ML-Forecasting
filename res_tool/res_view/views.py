from django.shortcuts import render
from .models import ResMeta, ResData, WeatherData
import pandas as pd
import plotly.express as px
from django.db.models import Avg
import datetime
from model_creation_scripts.data_preparation import prepare_dataset
from model_creation_scripts.modeling_utilities import *
import pickle
import os
from django.conf import settings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from .utils import train_and_evaluate_models

def home(request):
    reservoirs = ResMeta.objects.all()
    model_comparison_plot = None
    real_time_plot = None

    if request.method == "POST":
        selected_reservoir = request.POST.get('reservoir')
        training_end_date = '2022-12-31'
        test_start_date = '2023-01-01'
        lag_days = 14
        df_reservoir = pd.DataFrame(list(ResData.objects.filter(stn_id=selected_reservoir).values()))
        df_weather = pd.DataFrame(list(WeatherData.objects.filter(stn_id=selected_reservoir).values()))
        if 'stn_id_id' in df_weather.columns:
            df_weather.rename(columns={'stn_id_id': 'stn_id'}, inplace=True)
        
        if 'stn_id_id' in df_reservoir.columns:
            df_reservoir.rename(columns={'stn_id_id': 'stn_id'}, inplace=True)

        df_combined, predictor_columns = prepare_dataset(df_reservoir, df_weather, lag_days)        
        
        df_selected_reservoir = df_combined[df_combined['stn_id'] == selected_reservoir]     

        
        df_train = df_selected_reservoir[df_selected_reservoir['datetime'] <= training_end_date]
        X_train = df_train[predictor_columns]
        y_train= df_train['storage_value']
                                   
        # Split the data for the selected reservoir
        df_test = df_selected_reservoir[df_selected_reservoir['datetime'] >= test_start_date]

        # Create features and labels for training and testing datasets
        X_test = df_test[predictor_columns]
        y_test = df_test['storage_value']
        
        print(y_test.head())

        model_comparison_plot = train_and_evaluate_models(selected_reservoir,X_train, y_train, X_test, y_test, df_test)
        
        '''if selected_reservoir == '11122000':
            
            rf_santa_ynez = RandomForestRegressor(random_state=0)
            rf_santa_ynez.fit(X_train, y_train)
            
            svr_pipeline_santa_ynez = fit_svr(X_train, y_train, C=1000, epsilon=.85, gamma=0.01)
            
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
            mlp_regressor_santa_ynez.fit(X_train, y_train)

            knn_regressor_ynez = KNeighborsRegressor(n_neighbors=3) 
            knn_regressor_ynez.fit(X_train, y_train)
            
            kernel = (ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-2, 1e4)) * 
            RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) +  
            WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1e3))) 

            # GPR
            gpr_pipeline = make_pipeline(
                StandardScaler(),
                GaussianProcessRegressor(kernel=kernel, alpha=1e-3)  
            )
            gpr_pipeline_santa_ynez = gpr_pipeline.fit(X_train, y_train)
            
            #dt
            dt_pipeline = make_pipeline(
                StandardScaler(),
                DecisionTreeRegressor(random_state=0)
            )
            dt_regressor = DecisionTreeRegressor(
                max_depth=20,
                min_samples_leaf=1,
                min_samples_split=2,
                random_state=0
            )
            dt_regressor.fit(X_train, y_train)
            
            
            models = {
                'RandomForestRegressor': rf_santa_ynez,
                'SVR': svr_pipeline_santa_ynez,
                'MLPRegressor': mlp_regressor_santa_ynez,
                'KNeighborsRegressor': knn_regressor_ynez,
                'GaussianProcessRegressor': gpr_pipeline_santa_ynez,
                'DecisionTreeRegressor': dt_regressor,
            }
            
            # Run predictions for each model
            for model_name, model in models.items():
                y_pred = model.predict(X_test)
                predictions[model_name] = y_pred
                mae, mape = evaluate_model(y_test, y_pred)  # Pass y_test and y_pred to evaluate_model
                metrics[model_name] = (mae, mape)
                print(f'{model_name} MAE: {mae:.2f}, MAPE: {mape:.2f}%')
                
            model_comparison_plot = create_combined_plot(df_test, predictions, metrics)
                
                
        elif selected_reservoir == '11128300':
           
            rf_alisal = RandomForestRegressor(
                max_depth=44, 
                min_samples_leaf=1, 
                min_samples_split=2, 
                n_estimators=153, 
                random_state=0
            ) 
            
            rf_alisal.fit(X_train, y_train)
            
            svr_pipeline_alisal = fit_svr(X_train, y_train, C=1000, epsilon=0.1, gamma=0.01)
            
            mlp_regressor_alisal = MLPRegressor(
                activation='identity',
                alpha=0.0003962735057040824,
                hidden_layer_sizes=(150, 100),
                learning_rate='constant',
                max_iter=770,
                solver='lbfgs',
                random_state=42  # Adding a random state for reproducibility
            )
            mlp_regressor_alisal.fit(X_train, y_train)
            
            knn_regressor_alisal = KNeighborsRegressor(n_neighbors=3) 
            knn_regressor_alisal.fit(X_train, y_train)
            
            kernel = (ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-2, 1e4)) * 
            RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) +  
            WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1e3))) 
            # Create the GPR pipeline with the kernel and the alpha for numerical stability
            gpr_pipeline = make_pipeline(
                StandardScaler(),
                GaussianProcessRegressor(kernel=kernel, alpha=1e-3)  
            )
            gpr_pipeline_alisal = gpr_pipeline.fit(X_train, y_train)
            
            dt_pipeline = make_pipeline(
                StandardScaler(),
                DecisionTreeRegressor(random_state=0)
            )
            
            dt_regressor = DecisionTreeRegressor(
                max_depth=19,
                min_samples_leaf=1,
                min_samples_split=4,
                random_state=0
            )
            dt_regressor.fit(X_train, y_train)
            
            models = {
                'RandomForestRegressor': rf_alisal,
                'SVR': svr_pipeline_alisal,
                'MLPRegressor': mlp_regressor_alisal,
                'KNeighborsRegressor': knn_regressor_alisal,
                'GaussianProcessRegressor': gpr_pipeline_alisal,
                'DecisionTreeRegressor': dt_regressor,
            }
            # Run predictions for each model
            for model_name, model in models.items():
                y_pred = model.predict(X_test)
                predictions[model_name] = y_pred
                mae, mape = evaluate_model(y_test, y_pred)  # Pass y_test and y_pred to evaluate_model
                metrics[model_name] = (mae, mape)
                print(f'{model_name} MAE: {mae:.2f}, MAPE: {mape:.2f}%')
                model_comparison_plot = create_combined_plot(df_test, predictions, metrics)       
        '''
        real_time_plot = create_real_time_display_plot(df_test) # Create the real-time plot
        
    context = {
        'reservoirs': reservoirs,
        'model_comparison_plot': model_comparison_plot,
        'real_time_plot': real_time_plot,
    }
    return render(request, "res_view/home.html", context)

def create_real_time_display_plot(df_test):
    # Calculate the long-term daily average for 2023
    long_term_avg_2023 = df_test['storage_value'].mean()

    # Plot daily storage values for 2023
    fig = px.line(df_test, x='datetime', y='storage_value', title='Daily Water Storage in 2023')

    # Add long-term daily average for 2023 as a horizontal line
    fig.add_hline(y=long_term_avg_2023, line_dash="dot", 
                  annotation_text="2023 Long-term Average", 
                  annotation_position="bottom right")

    # Convert the plot to an HTML string
    plot_html = fig.to_html(full_html=False)

    return plot_html

def create_combined_plot(df_test, predictions, metrics):
    # Create the initial plot with actual data
    fig = px.line(df_test, x='datetime', y='storage_value', title='Predicted vs Actual Reservoir Storage')

    # Add the actual data line
    fig.add_scatter(x=df_test['datetime'], y=df_test['storage_value'], mode='lines', name='Actual')
    
    # Add each model's prediction line to the plot
    for model_name, y_pred in predictions.items():
        mae, mape = metrics[model_name]
        fig.add_scatter(x=df_test['datetime'], y=y_pred, mode='lines', 
                        name=f'{model_name} (MAE: {mae:.2f}, MAPE: {mape:.2f}%)')

    # Convert the plot to an HTML string
    plot_html = fig.to_html(full_html=False)

    return plot_html

def load_model(model_name):
    models_folder = settings.BASE_DIR / 'res_tool' / 'pkl_models'
    file_path = os.path.join(models_folder, model_name)
    if os.path.exists(file_path):
            print("Loading Trained Model")
            model = pickle.load(open(file_path, "rb"))
    else:
          print('No model with this name, check this and retry')
          model = None
    return model
# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return y_pred, rmse, mae, mape

def fit_svr(X_train, y_train, C=1000, epsilon=0.2, gamma=0.01):
    regr_pipeline = make_pipeline(StandardScaler(), SVR(C=C, epsilon=epsilon, gamma=gamma))
    regr_pipeline.fit(X_train, y_train)
    return regr_pipeline