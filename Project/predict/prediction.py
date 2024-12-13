from catboost import CatBoostRegressor
import pandas as pd


class PredictionModel:
    """
    A class for loading a pre-trained CatBoost regression model and making predictions on input data.

    This class provides the functionality to:
    - Load a trained CatBoost model from a specified file path.
    - Validate input data to ensure it matches the expected format and contains all necessary columns.
    - Make predictions using the trained model.

    Attributes:
        model (CatBoostRegressor): A pre-trained CatBoost regression model for making predictions.

    Methods:
        predict(input_df):
            Takes a DataFrame as input, validates it, and uses the model to predict house prices.
    """

    def __init__(self, model_path):
        """
        Initializes the PredictionModel class by loading the pre-trained CatBoost model.

        Args:
            model_path (str): The file path to the trained CatBoost model.

        Raises:
            ValueError: If the model cannot be loaded from the specified path.
        """
        try:
            # Initialize the CatBoost model
            self.model = CatBoostRegressor()
            self.model.load_model(model_path)
        except Exception as e:
            raise ValueError(f"Error loading the model: {e}")

    def predict(self, input_df):
        """
        Predicts the target value (house price) based on the input data.

        This method checks if the input data is in the correct format, ensuring that it is a pandas
        DataFrame with the expected columns. If the input is valid, the method makes a prediction
        using the loaded CatBoost model.

        Args:
            input_df (pd.DataFrame): A pandas DataFrame containing the feature values for prediction.
                                      The DataFrame must include the following columns:
                                      "Locality", "Zip code", "Property type", "Bedrooms",
                                      "Living area", "Surface of the plot", "Facades",
                                      "Building condition", "Fireplace", "Equipped kitchen",
                                      "Garden", "Garden surface", "Terrace", "Terrace surface",
                                      "Furnished", "Swimming pool", "Region".

        Returns:
            float: The predicted house price based on the input data.

        Raises:
            ValueError: If the input data is not a pandas DataFrame or if required columns are missing.
        """
        try:
            # Check if the input is a valid DataFrame with correct columns
            if not isinstance(input_df, pd.DataFrame):
                raise ValueError("Input should be a pandas DataFrame.")

            # Ensure that the DataFrame contains the expected columns
            expected_columns = [
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
            ]
            missing_columns = [
                col for col in expected_columns if col not in input_df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing value in input data: {', '.join(missing_columns)}"
                )

            # Make prediction
            prediction = self.model.predict(input_df)
            return prediction[0]  # Return predicted value

        except Exception as e:
            raise ValueError(f"Error during prediction: {e}")
