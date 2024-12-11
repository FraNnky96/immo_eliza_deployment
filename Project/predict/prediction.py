from catboost import CatBoostRegressor

class PredictionModel:
    def __init__(self, model_path):
        # Initialize the CatBoost model
        self.model = CatBoostRegressor()
        self.model.load_model(model_path)

    def predict(self, input_df):
        # Make prediction
        prediction = self.model.predict(input_df)
        return prediction[0]  # Return the predicted value