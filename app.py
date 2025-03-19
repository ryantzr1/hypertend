import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import json
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_iris import IRISVector
from langchain.chains import RetrievalQA

# Generate and save dummy data to a file (run only once)
def generate_and_save_dummy_data():
    start_date = datetime(2025, 3, 13, 0, 0)  # March 13, 2025, 00:00
    data = []
    
    for i in range(7 * 6):  # 7 days * 6 measurements per day = 42 entries
        timestamp = start_date + timedelta(hours=i * 4)
        
        base_weight = 70.0
        weight = round(base_weight + random.uniform(-0.5, 0.5), 1)
        
        base_pulse = 70
        pulse_rate = random.randint(base_pulse - 10, base_pulse + 10)
        
        systolic = random.randint(110, 140)
        diastolic = random.randint(70, 90)
        blood_pressure = f"{systolic}/{diastolic}"
        
        base_temp = 36.5
        temperature = round(base_temp + random.uniform(-1.5, 2.0), 1)
        if random.random() < 0.1:
            temperature += random.choice([-2.0, 2.0])
        
        spo2 = random.randint(95, 100)
        
        entry = {
            "timestamp": timestamp.strftime("%a %b %d %H:%M:%S %Y"),
            "weight": weight,
            "pulse_rate": pulse_rate,
            "blood_pressure": blood_pressure,
            "temperature": temperature,
            "spo2": spo2
        }
        data.append(entry)
    
    # Save to a file
    with open("sensor_data.json", "w") as f:
        json.dump(data, f)
    
    return data

# Load data from file or generate if not exists
@st.cache_data
def load_sensor_data():
    try:
        with open("sensor_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        data = generate_and_save_dummy_data()
        return data

# App setup
@st.cache_resource
def setup_app():
    if "OPENAI_API_KEY" not in os.environ:
        from dotenv import load_dotenv
        load_dotenv()
        if "OPENAI_API_KEY" not in os.environ:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or add it to your .env file.")
            st.stop()
    
    embeddings = OpenAIEmbeddings()
    
    hostname = "localhost"
    port = 1972
    namespace = "USER"
    username = "demo"
    password = "demo"
    connection_string = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"
    
    vector_store = IRISVector(
        embedding_function=embeddings,
        collection_name="sensor_data",
        connection_string=connection_string
    )
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    return embeddings, vector_store, llm, qa_chain

# Process data for visualization
def process_data(data):
    # Extract data for visualization
    timestamps = [entry["timestamp"] for entry in data]
    temperatures = [entry["temperature"] for entry in data]
    weights = [entry["weight"] for entry in data]
    pulse_rates = [entry["pulse_rate"] for entry in data]
    
    systolic_bps = []
    diastolic_bps = []
    for entry in data:
        systolic, diastolic = map(int, entry["blood_pressure"].split("/"))
        systolic_bps.append(systolic)
        diastolic_bps.append(diastolic)
    
    spo2s = [entry["spo2"] for entry in data]
    
    return {
        "timestamps": timestamps,
        "temperatures": temperatures,
        "weights": weights,
        "pulse_rates": pulse_rates,
        "systolic_bps": systolic_bps,
        "diastolic_bps": diastolic_bps,
        "spo2s": spo2s
    }

# Radial gauge visualization for temperature
def create_gauge(temperature):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=temperature,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Temperature (°C)"},
        gauge={
            'axis': {'range': [0, 50]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "lightblue"},
                {'range': [20, 35], 'color': "yellow"},
                {'range': [35, 50], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))
    return fig

# Timeline visualization
def create_timeline(data, x_values, y_values, title, yaxis_title):
    fig = go.Figure(data=[
        go.Scatter(x=x_values, y=y_values, mode='lines+markers')
    ])
    fig.update_layout(title=title, xaxis_title="Date/Time", yaxis_title=yaxis_title)
    return fig

# Convert timestamp strings to datetime objects
def parse_timestamps(timestamps):
    return [datetime.strptime(ts, "%a %b %d %H:%M:%S %Y") for ts in timestamps]

# Streamlit UI
def main():
    st.title("Smart Sensor Dashboard")
    st.write("Monitor your sensor data and query it with AI!")

    # Load data
    sensor_data = load_sensor_data()
    
    # Process data
    processed_data = process_data(sensor_data)
    
    # Parse timestamps for x-axis
    datetime_stamps = parse_timestamps(processed_data["timestamps"])
    
    # Setup app resources
    embeddings, vector_store, llm, qa_chain = setup_app()
    
    # First-time data storage
    if "data_stored" not in st.session_state:
        st.session_state.data_stored = False
    
    # Store data in vector store if not already done
    if not st.session_state.data_stored:
        with st.spinner("Processing sensor data for AI queries..."):
            for entry in sensor_data:
                text = (f"Temperature: {entry['temperature']}°C, "
                        f"Weight: {entry['weight']} kg, "
                        f"Pulse Rate: {entry['pulse_rate']} bpm, "
                        f"Blood Pressure: {entry['blood_pressure']} mmHg, "
                        f"SpO2: {entry['spo2']}%, "
                        f"at {entry['timestamp']}")
                
                vector = embeddings.embed_query(text)
                vector_store.add_texts([text])
            
            st.session_state.data_stored = True
            st.success("All sensor data processed and ready for querying!")
    
    # Date range filter
    st.sidebar.header("Filter Data")
    min_date = datetime_stamps[0].date()
    max_date = datetime_stamps[-1].date()
    
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    # Filter data based on date range
    filtered_indices = [i for i, dt in enumerate(datetime_stamps) if start_date <= dt.date() <= end_date]
    
    filtered_temperatures = [processed_data["temperatures"][i] for i in filtered_indices]
    filtered_weights = [processed_data["weights"][i] for i in filtered_indices]
    filtered_pulse_rates = [processed_data["pulse_rates"][i] for i in filtered_indices]
    filtered_systolic_bps = [processed_data["systolic_bps"][i] for i in filtered_indices]
    filtered_diastolic_bps = [processed_data["diastolic_bps"][i] for i in filtered_indices]
    filtered_spo2s = [processed_data["spo2s"][i] for i in filtered_indices]
    filtered_timestamps = [datetime_stamps[i] for i in filtered_indices]
    
    # Display summary statistics
    st.sidebar.header("Summary Statistics")
    if filtered_temperatures:
        st.sidebar.markdown(f"**Avg Temperature:** {sum(filtered_temperatures)/len(filtered_temperatures):.1f}°C")
        st.sidebar.markdown(f"**Avg Weight:** {sum(filtered_weights)/len(filtered_weights):.1f} kg")
        st.sidebar.markdown(f"**Avg Pulse Rate:** {sum(filtered_pulse_rates)/len(filtered_pulse_rates):.1f} bpm")
        st.sidebar.markdown(f"**Avg Blood Pressure:** {sum(filtered_systolic_bps)/len(filtered_systolic_bps):.1f}/{sum(filtered_diastolic_bps)/len(filtered_diastolic_bps):.1f} mmHg")
        st.sidebar.markdown(f"**Avg SpO2:** {sum(filtered_spo2s)/len(filtered_spo2s):.1f}%")
    
    # Temperature gauge for latest value
    if filtered_temperatures:
        st.subheader("Current Temperature")
        gauge = create_gauge(filtered_temperatures[-1])
        st.plotly_chart(gauge, use_container_width=True)
    
    # Display latest data
    if filtered_indices:
        latest_idx = filtered_indices[-1]
        latest_entry = sensor_data[latest_idx]
        st.subheader("Latest Data")
        st.info(f"Temperature: {latest_entry['temperature']}°C, "
                f"Weight: {latest_entry['weight']} kg, "
                f"Pulse Rate: {latest_entry['pulse_rate']} bpm, "
                f"Blood Pressure: {latest_entry['blood_pressure']} mmHg, "
                f"SpO2: {latest_entry['spo2']}%, "
                f"at {latest_entry['timestamp']}")
    
    # Create tabs for different charts
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Temperature", "Weight", "Pulse Rate", "Blood Pressure", "SpO2"])
    
    with tab1:
        if filtered_temperatures:
            st.plotly_chart(create_timeline(sensor_data, filtered_timestamps, filtered_temperatures, 
                                          "Temperature Timeline", "Temperature (°C)"), 
                          use_container_width=True)
    
    with tab2:
        if filtered_weights:
            st.plotly_chart(create_timeline(sensor_data, filtered_timestamps, filtered_weights, 
                                          "Weight Timeline", "Weight (kg)"), 
                          use_container_width=True)
    
    with tab3:
        if filtered_pulse_rates:
            st.plotly_chart(create_timeline(sensor_data, filtered_timestamps, filtered_pulse_rates, 
                                          "Pulse Rate Timeline", "Pulse Rate (bpm)"), 
                          use_container_width=True)
    
    with tab4:
        if filtered_systolic_bps:
            bp_fig = go.Figure()
            bp_fig.add_trace(go.Scatter(x=filtered_timestamps, y=filtered_systolic_bps, 
                                      mode='lines+markers', name='Systolic'))
            bp_fig.add_trace(go.Scatter(x=filtered_timestamps, y=filtered_diastolic_bps, 
                                      mode='lines+markers', name='Diastolic'))
            bp_fig.update_layout(title="Blood Pressure Timeline", 
                               xaxis_title="Date/Time", 
                               yaxis_title="Blood Pressure (mmHg)")
            st.plotly_chart(bp_fig, use_container_width=True)
    
    with tab5:
        if filtered_spo2s:
            st.plotly_chart(create_timeline(sensor_data, filtered_timestamps, filtered_spo2s, 
                                          "SpO2 Timeline", "SpO2 (%)"), 
                          use_container_width=True)
    
    # Query section
    st.header("Ask the AI")
    query = st.text_input("Enter your question (e.g., 'What's the average weight this week?')")
    if query:
        with st.spinner("Processing your query..."):
            response = qa_chain.run(query)
            st.write(f"**Response:** {response}")

if __name__ == "__main__":
    main()