from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from math import sqrt
import plotly.express as px

def train_and_evaluate_models(stn_id, X_train, y_train, X_test, y_test, df_test):
    if stn_id == '11122000':
        # Train and evaluate models for Santa Ynez Reservoir
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

        # dt

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

        # Dictionary to store predictions from each model
        predictions = {}
        metrics = {}

        # Run predictions for each model
        for model_name, model in models.items():
            y_pred, rmse, mae, mape = evaluate_model(model, X_test, y_test)  # Pass model, X_test, and y_test to evaluate_model
            predictions[model_name] = y_pred
            metrics[model_name] = (mae, mape)
            print(f'{model_name} MAE: {mae:.2f}, MAPE: {mape:.2f}%')

        model_comparison_plot = create_combined_plot(df_test, predictions, metrics)

    elif stn_id == '11128300':
        # Train and evaluate models for Alisal Reservoir
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

        # Dictionary to store predictions from each model
        predictions = {}
        metrics = {}

        # Run predictions for each model
        for model_name, model in models.items():
            y_pred, rmse, mae, mape = evaluate_model(model, X_test, y_test)  # Pass model, X_test, and y_test to evaluate_model
            predictions[model_name] = y_pred
            metrics[model_name] = (mae, mape)
            print(f'{model_name} MAE: {mae:.2f}, MAPE: {mape:.2f}%')
            model_comparison_plot = create_combined_plot(df_test, predictions, metrics)

    # You can return the necessary results here if needed
    return model_comparison_plot

def create_combined_plot(df_test, predictions, metrics):
    # Create the initial plot with actual data
    fig = px.line(df_test, x='datetime', y='storage_value', title='Predicted vs Actual Reservoir Storage')

    fig.add_scatter(x=df_test['datetime'], y=df_test['storage_value'], mode='lines', name='Actual')
    # Add each model's prediction line to the plot
    for model_name, y_pred in predictions.items():
        mae, mape = metrics[model_name]
        fig.add_scatter(x=df_test['datetime'], y=y_pred, mode='lines', 
                        name=f'{model_name} (MAE: {mae:.2f}, MAPE: {mape:.2f}%)')

    # Convert the plot to an HTML string
    plot_html = fig.to_html(full_html=False)

    return plot_html

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