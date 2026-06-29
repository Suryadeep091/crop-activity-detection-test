import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

def predict_from_pickle(model_path: str, input_df: pd.DataFrame, prediction_column: str = "prediction") -> pd.DataFrame:
    """
    Load a pickled model and make predictions on the input DataFrame,
    returning a new DataFrame with input features and predicted values.

    Parameters:
    - model_path: str, path to the pickle file containing the trained model and label encoder.
    - input_df: pd.DataFrame, input features for prediction.
    - prediction_column: str, name of the column to store predictions.

    Returns:
    - pd.DataFrame: a DataFrame combining input features and prediction column with actual labels.
    """
    # Load the model and label encoder from the pickle file
    with open(model_path, "rb") as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        label_encoder = saved_data['label_encoder']
        features = ['NDVI', 'EVI', 'RVI', 'crops']  # Define features in the desired order

    # Ensure input DataFrame has all required features
    missing_features = set(features) - set(input_df.columns)
    if missing_features:
        raise ValueError(f"Input DataFrame is missing required features: {missing_features}")

    # Store the date column
    date_column = input_df['date']

    # Reorder columns to match the expected feature order
    input_df = input_df[features]

    # Make predictions
    predictions_encoded = model.predict(input_df)
    
    # Convert encoded predictions back to original labels
    predictions = label_encoder.inverse_transform(predictions_encoded)

    # Combine input features with predictions
    result_df = input_df.copy()
    result_df['date'] = date_column  # Add back the date column
    result_df[prediction_column] = predictions

    # Reorder columns to put date first, followed by features and prediction
    result_df = result_df[['date'] + features + [prediction_column]]

    return result_df

def main():
    # Load your test data
    test_data = pd.read_csv('test_data.csv')  # Replace with your test data file
    
    # Make predictions
    model_path = 'model.pkl'  # Replace with your model file path
    predicted_df = predict_from_pickle(model_path, test_data, 'Predicted_Activity')
    
    # Calculate percentage of each activity
    activity_percentages = (predicted_df['Predicted_Activity'].value_counts(normalize=True) * 100).round(2)
    print("\nActivity Distribution (%):")
    print(activity_percentages)
    
    # Create time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(predicted_df['date'], predicted_df['Predicted_Activity'], marker='o', linestyle='-', markersize=4)
    plt.title('Agriculture Activity Over Time')
    plt.xlabel('Date')
    plt.ylabel('Activity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
