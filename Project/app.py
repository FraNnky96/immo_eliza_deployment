import streamlit as st
from predict.prediction import PredictionModel
from preprocessing.cleaning_data import Preprocessing

class PredictionApp:
    def __init__(self, scaler_path, model_path):
        # Initialize preprocessing and prediction classes
        self.preprocessing = Preprocessing(scaler_path)
        self.prediction_model = PredictionModel(model_path)

    def validate_inputs(self, user_inputs):
        if user_inputs["Living area"] <= 0:
            return "Living area must be a positive number."
        if not user_inputs["Locality"]:
            return "Locality must not be empty."
        if user_inputs["Bedrooms"] <= 0:
            return "Number of bedrooms must be at least 1."
        return None

    def display_input_form(self):
        st.title("House Price Prediction")

        st.write("""
            This application predicts house prices range based on various features.
            Enter the details below:
        """)

        # Input fields for user input (same as before)
        property_type = st.selectbox("Property type", ["House", "Apartment", "Villa", "Country Cottage", "Exceptional Property", "Mixed Use Building"])
        region = st.selectbox("Region", ["Brussels", "Wallonia", "Flanders"])
        locality = st.text_input("Locality")
        zip_code = st.number_input("Zip code", min_value=1000, max_value=9999, step=1)  # Restrict zip code range for Belgium
        bedrooms = st.number_input("Bedrooms", min_value=1, step=1)
        living_area = st.number_input("Living area in m²", min_value=1, step=1)
        surface_of_the_plot = st.number_input("Surface of the plot in m²", min_value=0, step=1)
        facades = st.number_input("Facades", min_value=1, step=1)
        building_condition = st.selectbox("Building condition", ["Just renovated", "As new", "Good", "To be done up", "To renovate"])
        fireplace = st.radio("Fireplace", ("No", "Yes"))
        equipped_kitchen = st.radio("Equipped Kitchen", ["No", "Yes"])
        garden = st.radio("Garden", ("No", "Yes"))
        garden_surface = st.number_input("Garden surface in m²", min_value=0, step=1) if garden == "Yes" else 0
        terrace = st.radio("Terrace", ("No", "Yes"))
        terrace_surface = st.number_input("Terrace surface in m²", min_value=0, step=1) if terrace == "Yes" else 0
        furnished = st.radio("Furnished", ("No", "Yes"))
        swimming_pool = st.radio("Swimming pool", ("No", "Yes"))

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

        return user_inputs

    def run(self):
        # Display the input form
        user_inputs = self.display_input_form()

        # Validate inputs
        error_message = self.validate_inputs(user_inputs)
        if error_message:
            st.error(error_message)
        else:
            # Process inputs for prediction
            processed_input = self.preprocessing.process(user_inputs)
            try:
                # Predict the price based on the processed input
                prediction = self.prediction_model.predict(processed_input)
                predicted_range_min = prediction * 0.90
                predicted_range_max = prediction * 1.10
                st.write(f"The predicted price range is: {predicted_range_min:,.2f}€ - {predicted_range_max:,.2f}€")
                st.write("This is the estimated price range based on the provided property features.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")


# Initialize the Streamlit app with paths to the model and scaler
if __name__ == "__main__":
    scaler_path = "Project/model/scaler.pkl"
    model_path = "Project/model/model.cbm"
    
    app = PredictionApp(scaler_path, model_path)
    app.run()


