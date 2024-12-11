from catboost import CatBoostRegressor


model = CatBoostRegressor()
model.load_model("model/model.cbm")

def predict(input_df): 
    # Make prediction
    prediction = model.predict(input_df)
    return prediction[0]  # Return the predicted value