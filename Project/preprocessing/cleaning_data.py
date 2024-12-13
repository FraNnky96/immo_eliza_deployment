import pandas as pd
import numpy as np
import joblib


class Preprocessing:
    def __init__(self, scaler_path):
        # Load the scaler used during training
        self.scaler = joblib.load(scaler_path)

    def process(self, user_inputs):
        # Preprocess input data (same as during training)
        input_df = pd.DataFrame(
            [user_inputs],
            columns=[
                "Locality",
                "Zip code",
                "Property type",
                "Bedrooms",
                "Living area",
                "Surface of the plot",
                "Facades",
                "Building condition",
                "Fireplace",
                "Equipped kitchen",
                "Garden",
                "Garden surface",
                "Terrace",
                "Terrace surface",
                "Furnished",
                "Swimming pool",
                "Region",
            ],
        )

        # Convert binary categorical variables to numerical (Yes -> 1, No -> 0)
        binary_columns = [
            "Fireplace",
            "Equipped kitchen",
            "Garden",
            "Terrace",
            "Furnished",
            "Swimming pool",
        ]
        for col in binary_columns:
            input_df[col] = input_df[col].apply(lambda x: 1 if x == "Yes" else 0)

        # Convert other categorical columns to string (if needed)
        categorical_columns = [
            "Building condition",
            "Region",
            "Property type",
            "Locality",
        ]
        for col in categorical_columns:
            input_df[col] = input_df[col].astype(str)

        # Handle missing values (example: fill missing garden surface and terrace surface with 0)
        input_df["Garden surface"] = input_df["Garden surface"].fillna(0)
        input_df["Terrace surface"] = input_df["Terrace surface"].fillna(0)

        # Scale the data (using the scaler fit on the training data)
        numeric_columns = input_df.select_dtypes(include=[np.number]).columns
        input_df[numeric_columns] = self.scaler.transform(input_df[numeric_columns])

        return input_df
