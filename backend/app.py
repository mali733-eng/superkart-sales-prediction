
# Import necessary libraries
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Keep the order of the keys in the JSON responses (e.g. the order of the rows in the batch predictions)
superkart_api.json.sort_keys = False

# Load the trained model (the preprocessing steps are part of the saved pipeline)
model = joblib.load("superkart_model.joblib")

# Features expected by the model
FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"


# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    data = request.get_json(silent=True)

    # Validate the input: it must be a JSON object containing all the features
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object'}), 400
    missing = [feature for feature in FEATURES if feature not in data]
    if missing:
        return jsonify({'error': f'Missing features: {missing}'}), 400

    # Extract relevant features from the input data
    sample = {feature: data[feature] for feature in FEATURES}

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    try:
        # Make a prediction using the trained model
        prediction = model.predict(input_data).tolist()[0]
    except Exception as e:
        return jsonify({'error': f'Unable to make the prediction: {e}'}), 400

    # Return the prediction as a JSON response
    return jsonify({'Sales': round(prediction, 2)})


# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files.get('file')
    if file is None:
        return jsonify({'error': "No file found in the request. Send the CSV file with the key 'file'"}), 400

    try:
        # Read the file into a DataFrame
        input_data = pd.read_csv(file)

        # Check that all the features are present in the file
        missing = [feature for feature in FEATURES if feature not in input_data.columns]
        if missing:
            return jsonify({'error': f'Missing columns in the CSV file: {missing}'}), 400

        # Make predictions for the batch data
        predictions = model.predict(input_data[FEATURES]).tolist()
    except Exception as e:
        return jsonify({'error': f'Unable to process the file: {e}'}), 400

    # Create an output dictionary mapping row index to predicted sales
    output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

    return jsonify(output_dict)


# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
