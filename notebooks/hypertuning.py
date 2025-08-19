from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestRegressor
import mlflow

# Set tracking URI and initialize MLflow autologging
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.autolog(silent=True)

# Define the search space for hyperparameters
space = {
    'n_estimators': hp.choice('n_estimators', range(10, 300)),
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10))
}

# Define objective function
def objective(params):
    with mlflow.start_run(nested=True):
        reg = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=42
        )
        
        reg.fit(X_train, y_train)
        r2_score = reg.score(X_val, y_val)

        return {'loss': -r2_score, 'status': STATUS_OK}

# Run optimization
trials = Trials()
with mlflow.start_run(run_name='hyperopt_optimization_regressor'):
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )

print("Best parameters found:", best)