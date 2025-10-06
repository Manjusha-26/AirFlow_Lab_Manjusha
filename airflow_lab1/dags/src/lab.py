import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import pickle
import os


def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.

    Returns:
        bytes: Serialized data.
    """
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)
    
    return serialized_data
    

def data_preprocessing(data):
    """
    Deserializes data, performs data preprocessing, and returns serialized processed data.

    Args:
        data (bytes): Serialized data to be deserialized and processed.

    Returns:
        bytes: Serialized preprocessed data (X_train, X_test, y_train, y_test).
    """
    df = pickle.loads(data)
    
    # Select only numeric columns for simplicity
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols].copy()
    
    # Remove Id column if exists
    if 'Id' in df_numeric.columns:
        df_numeric = df_numeric.drop('Id', axis=1)
    
    # Separate features and target
    if 'SalePrice' in df_numeric.columns:
        X = df_numeric.drop('SalePrice', axis=1)
        y = df_numeric['SalePrice']
    else:
        raise ValueError("SalePrice column not found in dataset")
    
    # Handle missing values with median imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Serialize the split data
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    serialized_data = pickle.dumps(processed_data)
    return serialized_data


def build_save_model(data, filename):
    """
    Builds a Random Forest Regressor model, saves it to a file, and returns test data.

    Args:
        data (bytes): Serialized preprocessed data.
        filename (str): Name of the file to save the model.

    Returns:
        dict: Dictionary containing test data for evaluation.
    """
    processed_data = pickle.loads(data)
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']
    
    # Build Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    
    # Save the trained model
    with open(output_path, 'wb') as f:
        pickle.dump(rf_model, f)
    
    print(f"Model saved to {output_path}")
    
    # Return test data for evaluation
    return pickle.dumps({'X_test': X_test, 'y_test': y_test})


def load_model_evaluate(filename, test_data):
    """
    Loads a saved Random Forest model and evaluates it on test data.

    Args:
        filename (str): Name of the file containing the saved model.
        test_data (bytes): Serialized test data.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    
    # Load the saved model
    loaded_model = pickle.load(open(output_path, 'rb'))
    print("Model loaded successfully!")
    
    # Load test data
    test_dict = pickle.loads(test_data)
    X_test = test_dict['X_test']
    y_test = test_dict['y_test']
    
    # Make predictions
    y_pred = loaded_model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print("="*50 + "\n")
    
    # Show sample predictions
    print("Sample Predictions vs Actual:")
    print("-" * 40)
    for i in range(min(5, len(y_test))):
        print(f"Predicted: ${y_pred[i]:,.2f} | Actual: ${y_test.iloc[i]:,.2f}")
    print("-" * 40)
    
    return {
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'predictions': y_pred[:5].tolist()
    }

def visualize_results(filename, test_data):
    """
    Creates visualizations for model results and feature importance.

    Args:
        filename (str): Name of the file containing the saved model.
        test_data (bytes): Serialized test data.

    Returns:
        str: Path to saved visualization.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for Docker
    import matplotlib.pyplot as plt
    
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, 'rb'))
    
    test_dict = pickle.loads(test_data)
    X_test = test_dict['X_test']
    y_test = test_dict['y_test']
    
    # Make predictions
    y_pred = loaded_model.predict(X_test)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Price ($)', fontsize=12)
    axes[0].set_ylabel('Predicted Price ($)', fontsize=12)
    axes[0].set_title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Feature Importance (Top 10)
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': loaded_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    axes[1].barh(feature_importance['feature'], feature_importance['importance'])
    axes[1].set_xlabel('Importance', fontsize=12)
    axes[1].set_ylabel('Feature', fontsize=12)
    axes[1].set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "model_visualization.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {plot_path}")
    return plot_path