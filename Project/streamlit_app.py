import streamlit as st
from predict.prediction import PredictionModel
from preprocessing.cleaning_data import Preprocessing

class PredictionApp:
    def __init__(self, model_path, scaler_path):
        # Initialize the Preprocessing and PredictionModel classes
        self.preprocessor = Preprocessing(scaler_path)
        self.model = PredictionModel(model_path)

    def validate_inputs(self, user_inputs):
        """Helper function for input validation"""
        if user_inputs["Living area"] <= 0:
            return "Living area must be a positive number."
        if not user_inputs["Locality"]:
            return "Locality must not be empty."
        if user_inputs["Bedrooms"] <= 0:
            return "Number of bedrooms must be at least 1."
        if user_inputs["Zip code"] < 1000 or user_inputs["Zip code"] > 9999:
            return "Zip code must be a 4-digit number."
        return None

    def run(self):
        # Set up the page config early
        st.set_page_config(layout="wide", page_title="House Price Prediction", page_icon=":house:")

        # Streamlit interface
        st.title("House Price Prediction")
        st.write("""
            This application predicts house prices based on various features.
            Enter the details below:
        """)

        # Input fields for user input
        property_type = st.selectbox("Property type", ["House", "Apartment", "Villa", "Country Cottage", "Exceptional Property", "Mixed Use Building"])
        region = st.selectbox("Region", ["Brussels", "Wallonia", "Flanders"])
        locality = st.text_input("Locality")
        zip_code = st.number_input("Zip code", min_value=1000, max_value=9999, step=1)  # Restrict zip code range for Belgium
        bedrooms = st.number_input("Bedrooms", min_value=1, step=1)
        living_area = st.number_input("Living area in m²", min_value=1, step=1)
        surface_of_the_plot = st.number_input("Surface of the plot in m²", min_value=0, step=1)
        facades = st.number_input("Facades", min_value=1, step=1)
        building_condition = st.selectbox("Building condition", ["Just renovated", "As new", "Good", "To be done up", "To renovate"])
        fireplace = st.radio("Fireplace", options=["No", "Yes"], horizontal=True)
        equipped_kitchen = st.radio("Equipped Kitchen", options=["No", "Yes"], horizontal=True)
        garden = st.radio("Garden", options=["No", "Yes"], horizontal=True)
        garden_surface = st.number_input("Garden surface in m²", min_value=0, step=1) if garden == "Yes" else 0
        terrace = st.radio("Terrace", options=["No", "Yes"], horizontal=True)
        terrace_surface = st.number_input("Terrace surface in m²", min_value=0, step=1) if terrace == "Yes" else 0
        furnished = st.radio("Furnished", options=["No", "Yes"], horizontal=True)
        swimming_pool = st.radio("Swimming pool", options=["No", "Yes"], horizontal=True)

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
            "Equipped kitchen": equipped_kitchen,
            "Garden": garden,
            "Garden surface": garden_surface,
            "Terrace": terrace,
            "Terrace surface": terrace_surface,
            "Furnished": furnished,
            "Swimming pool": swimming_pool,
            "Region": region
        }

        # Button to trigger prediction
        if st.button("Predict Price"):
            # Validate inputs
            error_message = self.validate_inputs(user_inputs)
            if error_message:
                st.error(error_message)
                return  # Stop execution if there's an error

            # Process inputs for prediction
            processed_input = self.preprocessor.process(user_inputs)
            try:
                # Predict the price based on the processed input
                prediction = self.model.predict(processed_input)
                st.write(f"The predicted price is: {prediction:,.2f}€")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# Instantiate and run the app
model_path = "./model/model.cbm"  # Update with your model path
scaler_path = "./model/scaler.pkl"  # Update with your scaler path
app = PredictionApp(model_path, scaler_path)
app.run()
