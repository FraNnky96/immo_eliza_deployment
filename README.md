# Immo Eliza Deployment

## Description

Deployment of the Immo Eliza project into a usable app with Streamlit.

- The `app.py` file creates the interface for the app and ensures proper formatting of values input by the user.
- The `cleaning_data.py` file formats the user input in the same way it was used during the model training in the Immo Eliza Regression project.
- The `prediction.py` file predicts the price based on the user's input.
- It also contains a `model` folder with the pre-trained model and scaler for data formatting.

## Installation

1. Create a virtual environment:

    ```bash
    python -m venv env
    ```

2. Install required libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

The application is already deployed at: [Link to app](https://immoelizadeployment-fl.streamlit.app/)

Or, for local hosting:

1. Install Streamlit:

    ```bash
    pip install streamlit
    ```

2. Run the application:

    ```bash
    streamlit run Project/app.py
    ```

## Visuals

![Screenshot 1](./visual/Application_visual1.PNG)
![Screenshot 2](./visual/Application_visual2.PNG)

## Personal Thoughts

Hello! This is the final part of the Red Line Project at BeCode.

Throughout the project, I faced several challenges—especially with the machine learning aspects, which I spent a significant amount of time trying to understand.

While the project is not perfect, here are some key points I learned:

- The dataset had its limitations, and I faced difficulties working with it.
- As I had no prior knowledge of machine learning, my focus was more on understanding the steps involved in the process rather than perfection. As a result, I spent a lot of time on minor issues that affected the overall state of the project.

That being said, this project taught me a lot, and I believe I won't make the same mistakes in the future!
