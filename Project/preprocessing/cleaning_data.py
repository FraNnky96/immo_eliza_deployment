import pandas as pd
import numpy as np
import joblib


class Preprocessing:
    """
    A class for preprocessing input data before making predictions.

    This class handles the necessary transformations of the user-provided input data, including:
    - Encoding binary categorical variables (e.g., Yes/No columns) to numerical values (1/0).
    - Converting other categorical columns to strings.
    - Handling missing values for specific columns.
    - Scaling numerical features using the pre-fitted scaler from the training phase.

    Attributes:
        scaler (object): A pre-loaded scaler (RobustScaler) used to scale numerical features.

    Methods:
        process(user_inputs):
            Processes the user input into a form suitable for model prediction, applying necessary transformations and scaling.
    """

    def __init__(self, scaler_path):
        """
        Initializes the Preprocessing object by loading the pre-trained scaler.

        Args:
            scaler_path (str): The file path to the saved scaler object used during model training.

        """

        self.scaler = joblib.load(scaler_path)

    def process(self, user_inputs):
        """
        Processes the user input data to prepare it for prediction.

        The input data is transformed in several steps:
        - Binary categorical columns ("Fireplace", "Equipped kitchen", "Garden", etc.) are encoded as 1 (Yes) or 0 (No).
        - Other categorical columns ("Building condition", "Region", "Property type", etc.) are converted to string type.
        - Missing values in columns like "Garden surface" and "Terrace surface" are filled with 0.
        - Numerical columns are scaled using the pre-fitted scaler loaded during initialization.

        Args:
            user_inputs (dict): A dictionary containing the user's input data where keys are feature names and values are feature values.

        Returns:
            pd.DataFrame: A DataFrame with preprocessed data, including numerical scaling, categorical encoding, and missing value handling.

        Raises:
            ValueError: If the input data does not contain the expected columns or the data format is invalid.

        """

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

        # Convert other categorical columns to string
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
