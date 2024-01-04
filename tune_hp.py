import argparse
import warnings
import pandas as pd
import optuna
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
# from tqdm.auto import tqdm
import joblib

# Suppress warnings
warnings.filterwarnings("ignore")

def training_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Model Hyperparameter Tuning')
    
    # Set data file paths
    parser.add_argument('--train_data_file', required=True, type=str, help="Path to the training data file")
    parser.add_argument('--val_data_file', required=True, type=str, help="Path to the validation data file")
    # Set number of trials for hyperparameter optimization
    parser.add_argument('--trial', default=100, type=int, help="Number of trials for Optuna optimization")
    parser.add_argument('--model_type', required=True, type=str, choices=['rf', 'xgb', 'mlp'], help="Type of the model")
    args = parser.parse_args()
    return args

def objective(trial, X_train, X_val, y_train, y_val, model_type):
    """
    Objective function for hyperparameter tuning with Optuna.
    Defines the parameter space and trains the model.
    """
    
    if model_type == "rf":
        # Define the hyperparameter configuration space

        param_grid = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "n_estimators": trial.suggest_int("n_estimators", 10, 500, step=50),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
        model = RandomForestClassifier(random_state=0, **param_grid)
        model.fit(X_train, y_train)
    
    elif model_type == "xgb":
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 100, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 100, log=True)
        }
        
        model = XGBClassifier(random_state=0,**param_grid)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, eval_metric="mlogloss")
    elif model_type == "mlp":
        param_grid = {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(32,), (64,), (32, 32), (32, 64), (64, 64)]),
            "activation": trial.suggest_categorical("activation", ["logistic", "relu"]),
            "solver": trial.suggest_categorical("solver", ["sgd", "adam"]),
            "alpha": trial.suggest_float("alpha", 1e-4, 1e-2, log=True),
            "early_stopping": True,
            "validation_fraction": 0.1,  # Fraction des données d'entraînement à utiliser comme ensemble de validation
            "n_iter_no_change": 10,  # Nombre d'itérations sans amélioration sur l'ensemble de validation
            "tol": trial.suggest_float("tol", 1e-4, 1e-3, log=True),
            "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
            "max_iter": trial.suggest_int("max_iter", 200, 1000, step=200)
        }

        model = MLPClassifier(random_state=0,**param_grid)
        model.fit(X_train, y_train)
    
    
    
#     # Initialize the Random Forest model
#     model = RandomForestClassifier(random_state=0, **param_grid)
    
    # Fit the model
    # model.fit(X_train, y_train)

    # Make predictions on the validation set
    preds = model.predict_proba(X_val)
    
    # Return the log loss (the metric to be minimized)
    return log_loss(y_val, preds)

def find_hp_opt(trial, X_train, X_val, y_train, y_val, optim='minimize', study_name='Random Forest Classifier', model_type="rf"):
    """
    Function to perform hyperparameter optimization using Optuna.
    """
    study = optuna.create_study(direction=optim, study_name=study_name)
    func = lambda trial: objective(trial, X_train, X_val, y_train, y_val, model_type)
    study.optimize(func, n_trials=trial)
    return study

if __name__ == "__main__":
    
    args = training_args()
    
    # Load and preprocess data
    print('Load and preprocess data...', end=' ')

    # Load the TF-IDF Vectorizer
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    # Load the LabelEncoder 
    label_encoder = joblib.load('models/label_encoder.pkl')
    # Load the training data
    train_data = pd.read_csv(args.train_data_file)
    # Transform train data
    X_train = tfidf_vectorizer.transform(train_data["cleaned_resume"])
    # Transform train labels
    y_train = label_encoder.transform(train_data["class"])

    # Load the validation data
    val_data = pd.read_csv(args.val_data_file)
    # Transform validation data and labels
    X_val = tfidf_vectorizer.transform(val_data["cleaned_resume"])
    y_val = label_encoder.transform(val_data["class"])

    print('Done.')
    
    # Perform hyperparameter tuning
    study = find_hp_opt(args.trial, X_train, X_val, y_train, y_val, optim='minimize', study_name=args.model_type + ' Classifier', model_type=args.model_type)
    
    # Save the configuration file with the best parameters
    print('Save config file...', end=' ')
    with open(f"config_hp_{args.model_type}.pkl", "wb") as file:
        pickle.dump(study.best_params, file)
    print('Done.')
