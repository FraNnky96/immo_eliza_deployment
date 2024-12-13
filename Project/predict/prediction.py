from catboost import CatBoostRegressor
import pandas as pd


class PredictionModel:
    def __init__(self, model_path):
        try:
            # Initialize the CatBoost model
            self.model = CatBoostRegressor()
            self.model.load_model(model_path)
        except Exception as e:
            raise ValueError(f"Error loading the model: {e}")

    def predict(self, input_df):
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
