import streamlit as st
import joblib
import pandas as pd

# Load the model and scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('preprocessork.pkl')

# Define the app layout
st.title("Credit Card Customer Clustering App")
st.markdown("---")
st.write("This Streamlit web application utilizes a K-means clustering algorithm to analyze credit card customer data and predict the cluster to which each customer belongs.")
# Create input fields for user input
st.sidebar.header('User Input Parameters')
st.markdown("---")
def user_input_features():
    avg_credit_limit = st.sidebar.number_input('Average Credit Limit', min_value=0, value=100000)
    total_credit_cards = st.sidebar.number_input('Total Credit Cards', min_value=0, value=2)
    total_visits_bank = st.sidebar.number_input('Total Visits to Bank', min_value=0, value=1)
    total_visits_online = st.sidebar.number_input('Total Visits Online', min_value=0, value=1)
    total_calls_made = st.sidebar.number_input('Total Calls Made', min_value=0, value=0)
    st.markdown("---")
    # Create a DataFrame with the input values
    user_input = pd.DataFrame({
        'Avg_Credit_Limit': [avg_credit_limit],
        'Total_Credit_Cards': [total_credit_cards],
        'Total_visits_bank': [total_visits_bank],
        'Total_visits_online': [total_visits_online],
        'Total_calls_made': [total_calls_made]
    })
    
    return user_input

user_input = user_input_features()

# Preprocess user input
scaled_input = scaler.transform(user_input)

# Predict the cluster
cluster_prediction = kmeans.predict(scaled_input)

# Display results
st.subheader('Cluster Prediction')
st.write(f'The customer belongs to cluster: {cluster_prediction[0]}')
st.markdown("---")
# Cluster descriptions
st.write("""
**Cluster Descriptions:**
- Cluster 0 : Customers who have a total credit amount and a medium average credit limit, meaning they are likely to have a moderate financial profile.
- Cluster 1 : Customers who have the lowest total credit and average limit.suggesting these customers may be less financially active or have lower creditworthiness.
- Cluster 2 : Customers who have the highest total credit cards and average limits.These customers are likely to be more financially engaged and potentially have a stronger credit profile.
""")
st.markdown("---")
# Optionally display the raw data
st.subheader('User Input Data')
st.write(user_input)
st.markdown("---")
