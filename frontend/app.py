
import io
import os
import pandas as pd
import requests
import streamlit as st

# Base URL of the Flask backend
# Inside the docker network the backend container is reached using its container name ("backend")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")

# Page title
st.set_page_config(page_title="SuperKart System")
st.title("SuperKart System")
st.write(
    "Enter the product and store details below to predict the total sales of the product in the store."
)

# Input fields for product data
st.subheader("Product details")
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input(
    "Product Allocated Area", min_value=0.0, max_value=1.0, value=0.027, step=0.001, format="%.3f"
)
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
Product_Id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"])
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"], index=1)

# Input fields for store data
st.subheader("Store details")
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"], index=1)
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"], index=1)
Store_Type = st.selectbox(
    "Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"], index=1
)
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)

# Create JSON payload
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category,
}

# Single Prediction
if st.button("Predict", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=30)

        if response.status_code == 200:
            result = response.json()
            predicted_sales = result["Sales"]
            st.success(f"Predicted Product Store Sales Total: {predicted_sales:,.2f}")
        else:
            st.error(f"The prediction API returned an error ({response.status_code}): {response.text}")
    except requests.exceptions.RequestException:
        st.error("Unable to connect to the prediction API.")

# Batch Prediction
st.subheader("Batch Prediction")
st.write(
    "Upload a CSV file containing the columns Product_Weight, Product_Sugar_Content, Product_Allocated_Area, "
    "Product_MRP, Store_Size, Store_Location_City_Type, Store_Type, Product_Id_char, Store_Age_Years and "
    "Product_Type_Category."
)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    if st.button("Predict for Batch", type="primary"):
        try:
            # Content of the uploaded file (used for the API call and to display the results)
            file_bytes = uploaded_file.getvalue()

            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": (uploaded_file.name, file_bytes, "text/csv")},
                timeout=60,
            )

            if response.status_code == 200:
                results = response.json()
                st.success("Predictions completed successfully!")

                # Display the uploaded rows along with their predicted sales
                batch_df = pd.read_csv(io.BytesIO(file_bytes))
                batch_df["Predicted_Sales"] = [results.get(str(i)) for i in range(len(batch_df))]
                st.dataframe(batch_df, use_container_width=True)

                # Download of the predictions as a CSV file (with on_click="ignore" the application is not re-run
                # by the download, so the table stays on the screen)
                st.download_button(
                    "Download predictions as CSV",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="superkart_batch_predictions.csv",
                    mime="text/csv",
                    on_click="ignore",
                )
            else:
                st.error(f"The prediction API returned an error ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException:
            st.error("Unable to connect to the prediction API.")
