import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


def setup_logging(log_level=logging.INFO):
    """Sets up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

def load_data(data_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully from {data_path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(df, target_column):
    """Preprocesses the data, splitting into features and target."""
    try:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        logging.info("Data preprocessed: Features and target separated.")
        return X, y
    except KeyError:
        logging.error(f"Target column '{target_column}' not found in the data.")
        raise

def train_model(X_train, y_train, model_type='logistic_regression', **kwargs):
    """Trains a machine learning model."""
    try:
        if model_type == 'logistic_regression':
            model = LogisticRegression(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        logging.info(f"Model trained in {training_time:.2f} seconds.")
        return model, training_time
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test)
        logging.info(f"Model accuracy: {accuracy:.4f}")
        logging.info(f"Classification Report:\n{report}")
        return accuracy, report
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def save_model(model, model_path):
    """Saves the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main():
    """Main function to orchestrate the model training and evaluation."""
    parser = argparse.ArgumentParser(description="Train and evaluate a machine learning model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file.")
    parser.add_argument("--target_column", type=str, required=True, help="Name of the target column.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data for testing.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level.")
    parser.add_argument("--model_type", type=str, default="logistic_regression", choices=["logistic_regression"], help="Type of model to train.")
    parser.add_argument("--solver", type=str, default="liblinear", help="Solver for Logistic Regression.")

    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper()))

    try:
        df = load_data(args.data_path)
        X, y = preprocess_data(df, args.target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

        model, training_time = train_model(X_train, y_train, model_type=args.model_type, solver=args.solver, random_state=args.random_state)
        accuracy, report = evaluate_model(model, X_test, y_test)

        save_model(model, args.model_path)

        logging.info("Training and evaluation completed successfully.")

    except Exception as e:
        logging.critical(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()