import streamlit as st
from predict.prediction import predict
from preprocessing.cleaning_data import preprocess

# Streamlit interface
st.title("House Price Prediction")

st.write("""
    This application predicts house prices range based on various features.
    Enter the details below:
""")

# Input fields for user input
property_type = st.selectbox("Property type", ["House", "Apartment", "Villa", "Country Cottage", "Exceptional Property", "Mixed Use Building"])
region = st.selectbox("Region", ["Brussels", "Wallonia", "Flanders"])
locality = st.text_input("Locality")
zip_code = st.number_input("Zip code", min_value=0, step=1)
bedrooms = st.number_input("Bedrooms", min_value=1, step=1)
living_area = st.number_input("Living area in m²", min_value=1, step=1)
surface_of_the_plot = st.number_input("Surface of the plot in m²", min_value=1, step=1)
facades = st.number_input("Facades", min_value=1, step=1)
building_condition = st.selectbox("Building condition", ["Just renovated", "As new", "Good", "To be done up", "To renovate"])
fireplace = st.radio("Fireplace", ("Yes", "No"))
equiped_kitchen = st.radio("Equipped Kitchen", ["Yes", "No"])
garden = st.radio("Garden", ("Yes", "No"))
garden_surface = st.number_input("Garden surface in m²", min_value=1, step=1) if garden == "Yes" else 0
terrace = st.radio("Terrace", ("Yes", "No"))
terrace_surface = st.number_input("Terrace surface in m²", min_value=1, step=1) if terrace == "Yes" else 0
furnished = st.radio("Furnished", ("Yes", "No"))
swimming_pool = st.radio("Swimming pool", ("Yes", "No"))

# Collect all input data
user_inputs = {
    "Locality": locality,
    "Zip code": zip_code,
    "Property type": property_type,
    "Bedrooms": bedrooms,
    "Living area": living_area,
    "Surface of the plot": surface_of_the_plot,
    "Facades": facades,
    "Building condition": building_condition,
    "Fireplace": fireplace,
    "Equipped kitchen": equiped_kitchen,
    "Garden": garden,
    "Garden surface": garden_surface,
    "Terrace": terrace,
    "Terrace surface": terrace_surface,
    "Furnished": furnished,
    "Swimming pool": swimming_pool,
    "Region": region
}

# When the user clicks the button, make a prediction
if st.button("Predict Price"):
    # Validate inputs
    if living_area <= 0:
        st.error("Living area must be a positive number.")
    elif surface_of_the_plot <= 0:
        st.error("Surface of the plot must be a positive number.")
    elif bedrooms <= 0:
        st.error("Number of bedrooms must be at least 1.")
    else:
        # Process inputs for prediction
        processed_input = preprocess(user_inputs)
        try:
            # Predict the price based on the processed input
            prediction = predict(processed_input)
            st.write(f"The predicted price range is: {prediction*0.85:,.2f} - {prediction:,.2f}€")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
