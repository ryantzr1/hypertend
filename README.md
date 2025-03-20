# HealthHub Dashboard

A comprehensive health monitoring dashboard with real-time data visualization and AI-powered insights using LangChain and InterSystems IRIS.

## Features

- Real-time health data monitoring and visualization
- Historical health data tracking with customizable date ranges
- Vector database storage with InterSystems IRIS
- AI-powered natural language queries using LangChain and OpenAI
- Interactive charts and gauges for vital signs

## Data Tracked

- Body temperature (°C)
- Weight (kg)
- Pulse rate (bpm)
- Blood pressure (mmHg)
- SpO2 levels (%)

## Requirements

- Python 3.11 or higher
- InterSystems IRIS database
- OpenAI API key

## Setup

### Environment Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/ryantzr1/healthhub.git
   cd healthhub
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install streamlit pandas plotly langchain-community langchain-openai langchain-iris langchain python-dotenv
   ```

4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

### InterSystems IRIS Setup

You can quickly set up IRIS using Docker:

```bash
docker run -d --name iris-comm -p 1972:1972 -p 52773:52773 -e IRIS_PASSWORD=demo -e IRIS_USERNAME=demo intersystemsdc/iris-community:latest
```

## Usage

1. Start the application:

   ```bash
   ./venv/bin/streamlit run app.py
   ```

2. The dashboard will open in your browser at http://localhost:8501

3. Use the date range selector in the sidebar to filter data

4. View summary statistics in the sidebar

5. Explore different health metrics using the tabbed interface

6. Ask questions in natural language in the "Ask the AI" section, such as:
   - "What was my average temperature last week?"
   - "Was there any abnormal blood pressure reading?"
   - "Show me the trend in my pulse rate"
   - "When did I have the highest SpO2 reading?"

## Development

The application is structured as follows:

- `app.py` - Main Streamlit application with dashboard UI and data processing
- `.env` - Environment variables (API keys)
- `requirements.txt` - Python dependencies

### Requirements.txt

```
streamlit==1.32.0
langchain==0.1.4
langchain-community==0.0.16
langchain-openai==0.0.5
langchain-iris==0.0.1
intersystems-iris==1.0.0
openai==1.12.0
python-dotenv==1.0.1
plotly==5.18.0
pandas==2.1.4
```

## Troubleshooting

- **IRIS Connection Issues**: Ensure the InterSystems IRIS container is running: `docker ps`
- **OpenAI API Errors**: Verify your API key is valid and has sufficient quota
- **Missing dependencies**: Make sure you've activated the virtual environment and installed all packages: `pip install -r requirements.txt`
- **Embedding errors**: Check that your vector store initialization is correct in `setup_app()`
- **Data not displaying**: Ensure the JSON data file exists or is being generated correctly

## License

MIT

## Acknowledgements

- InterSystems IRIS for vector database
- LangChain for AI framework
- OpenAI for embeddings and completions
- Streamlit for the interactive dashboard

---

**Note**: This project uses mock health data for demonstration purposes. In a real-world application, appropriate data security, privacy measures, and medical device certifications would be required.
