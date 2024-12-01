# New code
import streamlit as st
import snowflake.connector
import pandas as pd
import openai
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Connect to Snowflake
def get_snowflake_connection():
    conn = snowflake.connector.connect(
        user='JANESH',  # Your Snowflake username
        password='Janeshkj@2025',  # Your Snowflake password
        account='nu73073.central-india.azure',  # Correct account identifier
        warehouse='COMPUTE_WH',
        database='INFRASTRUCTURE_DB',
        schema='PUBLIC'
    )
    return conn

# Function to run queries
def run_query(query):
    conn = get_snowflake_connection()
    cur = conn.cursor()
    cur.execute(query)
    df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    cur.close()
    conn.close()
    return df

# Define the function to generate insights
def generate_insights(context):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Based on this public transport data, provide insights:\n\n{context}"}],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message['content'].strip()

# Function to generate predictions
def generate_predictions(data, target_column):
    df = data.copy()
    df = df.dropna()  # Remove missing data
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict(X[-5:])
    return prediction

# Streamlit App Layout
st.title("Public Transport Insights Dashboard")
st.subheader("Comprehensive Analysis of Transport Data")

# Sidebar for Navigation
st.sidebar.title("Navigation")
datasets = [
    "Basic Fare of Passenger Trains",
    "SRTU Fleet Data",
    "SRTU Financial Data",
    "Transport Vehicles Data",
    "State Revenue Analysis",
    "Commercial Vehicles Analysis",
    "Vehicle Registrations Over Years"
]
option = st.sidebar.selectbox("Choose a dataset to explore", datasets)

# Initialize a variable to store the context data for insights
context_data = None

# Data Display and Visualization
if option == "Basic Fare of Passenger Trains":
    st.header("Basic Fare of Passenger Trains")
    query = "SELECT * FROM BASICFAREOFPASS LIMIT 10"
    fare_data = run_query(query)
    st.write(fare_data)

    st.subheader("Fare Comparison by Distance")
    st.bar_chart(fare_data.set_index("DISTANCES")[["BASIC_FARE_ORDINARY_PASSENGER_TRAINS", "BASIC_FARE_MAIL_EXPRESS_TRAINS"]])
    context_data = fare_data

elif option == "SRTU Fleet Data":
    st.header("SRTU Fleet Data - Fleet Size, Utilization, and Efficiency")
    query = "SELECT * FROM SRTU1617_A1 LIMIT 10"
    fleet_data = run_query(query)
    st.write(fleet_data)

    st.subheader("Fleet Utilization by SRTUs (2015-16 vs 2016-17)")
    st.line_chart(fleet_data.set_index("SRTU_NAME")[["FLEET_UTILISATION_2016_17", "FLEET_UTILISATION_2015_16"]])
    context_data = fleet_data

elif option == "SRTU Financial Data":
    st.header("SRTU Financial Data - Revenue, Cost, and Profit Analysis")
    query = "SELECT * FROM SRTU1617_A2 LIMIT 10"
    financial_data = run_query(query)
    st.write(financial_data)

    st.subheader("Yearly Revenue Trends for SRTUs")
    st.line_chart(financial_data.set_index("SRTU_NAME")[["TOTAL_TRAFFIC_REVENUE_2016_17", "TOTAL_TRAFFIC_REVENUE_2015_16"]])

    st.subheader("Revenue Distribution by SRTUs (2016-17)")
    fig, ax = plt.subplots()
    financial_data.set_index("SRTU_NAME")["TOTAL_TRAFFIC_REVENUE_2016_17"].plot.pie(
        autopct='%1.1f%%', ax=ax, startangle=90
    )
    ax.set_ylabel('')
    ax.set_title('Revenue Share by SRTU (2016-17)', fontsize=14)
    st.pyplot(fig)
    context_data = financial_data

elif option == "Transport Vehicles Data":
    st.header("Transport Vehicles Data")
    query = "SELECT * FROM RTYB1819_A3_4C LIMIT 10"
    transport_data = run_query(query)
    st.write(transport_data)

    st.subheader("Scooter Distribution Across States")
    st.bar_chart(transport_data.set_index("STATES_UNION_TERRITORIES")[["TWO_WHEELERS_I_SCOOTERS"]])

    st.subheader("Scatter Plot: Two-Wheelers vs Cars")
    st.scatter_chart(transport_data.set_index("STATES_UNION_TERRITORIES")[["TWO_WHEELERS_I_SCOOTERS", "CARS_II"]])
    context_data = transport_data

elif option == "State Revenue Analysis":
    st.header("State Revenue Analysis")
    query = "SELECT * FROM RTYB1819_A5_3 LIMIT 10"
    state_revenue = run_query(query)
    st.write(state_revenue)

    st.subheader("Revenue Trends (2017-18 vs 2018-19)")
    st.line_chart(state_revenue.set_index("NAME_OF_STATE_UT")[["YEAR_2017_18_ACCOUNTS_TAX_ON_VEHICLES", "YEAR_2018_19_ACCOUNTS_TAX_ON_VEHICLES"]])

    st.subheader("Tax Distribution (2017-18)")
    fig, ax = plt.subplots()
    state_revenue.set_index("NAME_OF_STATE_UT")["YEAR_2017_18_ACCOUNTS_TAX_ON_VEHICLES"].plot.pie(
        autopct='%1.1f%%', ax=ax, startangle=90
    )
    ax.set_ylabel('')
    ax.set_title('Tax Distribution by States (2017-18)', fontsize=14)
    st.pyplot(fig)
    context_data = state_revenue

elif option == "Commercial Vehicles Analysis":
    st.header("Commercial Vehicles Analysis")
    query = "SELECT * FROM RTYB1819_A3_8 LIMIT 10"
    commercial_data = run_query(query)
    st.write(commercial_data)

    st.subheader("Commercial Vehicles Trends")
    st.line_chart(commercial_data.set_index("STATES_UNION_TERRITORIES")[["STAGE_CARRIAGE_PUBLIC_STU", "CONTRACT_CARRIAGES_E_RICKSHAW", "CONTRACT_CARRIAGES_OMNI_BUS_SP"]])
    context_data = commercial_data

elif option == "Vehicle Registrations Over Years":
    st.header("Vehicle Registrations Over Years")
    query = "SELECT * FROM RTYB1819_A3_1 LIMIT 10"
    registrations_data = run_query(query)
    st.write(registrations_data)

    st.subheader("Vehicle Registrations Trends (2017-18 vs 2018-19)")
    st.line_chart(registrations_data.set_index("STATES_UTS")[["TOTAL_REGISTERED_MOTOR_VEHICLES_2017_18", "TOTAL_REGISTERED_MOTOR_VEHICLES_2018_19"]])

    st.subheader("Registration Distribution (2017-18)")
    fig, ax = plt.subplots()
    registrations_data.set_index("STATES_UTS")["TOTAL_REGISTERED_MOTOR_VEHICLES_2017_18"].plot.pie(
        autopct='%1.1f%%', ax=ax, startangle=90
    )
    ax.set_ylabel('')
    ax.set_title('Vehicle Registrations (2017-18)', fontsize=14)
    st.pyplot(fig)
    context_data = registrations_data

# Generate Insights and Predictions Button
if st.button("Generate Insights and Recommendations"):
    if context_data is not None:
        insights = generate_insights(context_data.to_string(index=False))
        st.subheader("Generated Insights")
        st.write(insights)

        # st.write("Based on this data, here are some predictions:")
        if 'TOTAL_REGISTERED_MOTOR_VEHICLES_2018_19' in context_data.columns:
            st.subheader("Predictions and Recommendations")
            predictions = generate_predictions(context_data, 'TOTAL_REGISTERED_MOTOR_VEHICLES_2018_19')
            st.write(f"Predicted values for the next entries: {predictions}")
        # else:
        #     st.write("No numerical target column available for prediction.")
    else:
        st.write("No data available to generate insights and predictions!")

# Footer
st.write("Powered by Snowflake and Streamlit for a data-driven transport future.")