import pandas as pd
import numpy as np
import joblib 

# Load the scaler used during training
scaler = joblib.load('Project/model/scaler.pkl')

def preprocess(user_inputs):
    # Preprocess input data (same as during training)
    input_df = pd.DataFrame([user_inputs], columns=['Locality', 'Zip code', 'Property type', 'Bedrooms', 'Living area', 'Surface of the plot', 'Facades', 'Building condition', 'Fireplace', 'Equipped kitchen', 'Garden', 'Garden surface', 'Terrace', 'Terrace surface', 'Furnished', 'Swimming pool', 'Region'])
    
    # Convert categorical variables to numerical
    input_df["Fireplace"] = input_df["Fireplace"].apply(lambda x: 1 if x == "Yes" else 0)
    input_df["Equipped kitchen"] = input_df["Equipped kitchen"].apply(lambda x: 1 if x == "Yes" else 0)
    input_df["Garden"] = input_df["Garden"].apply(lambda x: 1 if x == "Yes" else 0)
    input_df["Terrace"] = input_df["Terrace"].apply(lambda x: 1 if x == "Yes" else 0)
    input_df["Furnished"] = input_df["Furnished"].apply(lambda x: 1 if x == "Yes" else 0)
    input_df["Swimming pool"] = input_df["Swimming pool"].apply(lambda x: 1 if x == "Yes" else 0)
    
    # Convert other categorical columns as required
    input_df["Building condition"] = input_df["Building condition"].astype(str)
    input_df["Region"] = input_df["Region"].astype(str)
    input_df["Property type"] = input_df["Property type"].astype(str)
    input_df["Locality"] = input_df["Locality"].astype(str)

    # Scale the data (using the scaler fit on the training data)
    numeric_columns = input_df.select_dtypes(include=[np.number]).columns
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

    return input_df
    