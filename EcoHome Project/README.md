# EcoHome Energy Advisor

An AI-powered energy optimization agent that helps customers reduce electricity costs and environmental impact through personalized recommendations.

## Project Overview

EcoHome is a smart-home energy start-up that helps customers with solar panels, electric vehicles, and smart thermostats optimize their energy usage. The Energy Advisor agent provides personalized recommendations about when to run devices to minimize costs and carbon footprint.

### Key Features

- **Weather Integration**: Uses weather forecasts to predict solar generation
- **Dynamic Pricing**: Considers time-of-day electricity prices for cost optimization
- **Historical Analysis**: Queries past energy usage patterns for personalized advice
- **RAG Pipeline**: Retrieves relevant energy-saving tips and best practices
- **Multi-device Optimization**: Handles EVs, HVAC, appliances, and solar systems
- **Cost Calculations**: Provides specific savings estimates and ROI analysis

## Project Structure

```
EcoHome Project/
├── data/
│   ├── documents/
│   ├── energy_data.db
│   └── vectorstore/
├── evaluation/
│   ├── __init__.py
│   └── report_generator.py
├── models/
│   ├── __init__.py
│   └── energy.py
├── prompts/
│   └── system_prompt.txt
├── agent.py
├── demo_weather_forecast.py
├── smoke_test.py
├── test_weather_forecast.py
├── tools.py
├── requirements.txt
├── 01_db_setup.ipynb
├── 02_rag_setup.ipynb
├── 03_run_and_evaluate.ipynb
└── README.md
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_vocareum_or_openai_compatible_key_here
OPENAI_BASE_URL=https://openai.vocareum.com/v1
```

If you are using Vocareum, keep the `OPENAI_BASE_URL` value shown above. The agent and RAG pipeline are configured to use an OpenAI-compatible endpoint.

### 3. Run the Notebooks

Execute the notebooks in order:

1. **01_db_setup.ipynb** - Set up the SQLite database and populate sample energy and solar data
2. **02_rag_setup.ipynb** - Build or load the Chroma knowledge base for energy advice
3. **03_run_and_evaluate.ipynb** - Run end-to-end scenarios and generate the evaluation report

### 4. Optional Script Checks

You can also run the lightweight Python scripts for local validation:

```bash
python smoke_test.py
python test_weather_forecast.py
python demo_weather_forecast.py
```

## Runtime Environment

- Python: 3.14.5
- LangChain: 1.3.1
- langchain-openai: 1.2.2
- langchain-community: 0.4.2
- LangGraph: 1.2.1
- langchain-chroma: 1.1.0
- langchain-text-splitters: 1.1.2
- ChromaDB: 1.5.9
- OpenAI client: 2.38.0
- SQLAlchemy: 2.0.49
- python-dotenv: 1.2.2
- pandas: 3.0.3
- numpy: 2.4.6
- pytest: 9.0.3
- requests: 2.34.2

Exact package pins are included in [requirements.txt](c:/Users/BharathKumar/OneDrive%20-%20Synapx/lcgraph/EcoHome%20Project/requirements.txt).

## Knowledge Base Coverage

The RAG corpus includes the original starter tips plus five additional domain documents covering:

- HVAC optimization strategies
- Smart home automation tips
- Renewable energy integration
- Seasonal energy management
- Energy storage optimization

Current document set includes:

- `tip_device_best_practices.txt`
- `tip_energy_savings.txt`
- `hvac_optimization_guide.txt`
- `advanced_hvac_optimization_strategies.txt`
- `smart_home_automation_guide.txt`
- `renewable_energy_integration_playbook.txt`
- `seasonal_energy_management_checklist.txt`
- `battery_storage_optimization_guide.txt`

## Agent Capabilities

### Tools Available

- **Weather Forecast**: Get hourly weather predictions and solar irradiance
- **Electricity Pricing**: Access time-of-day pricing data
- **Energy Usage Query**: Retrieve historical consumption data
- **Solar Generation Query**: Get past solar production data
- **Energy Tips Search**: Find relevant energy-saving recommendations
- **Savings Calculator**: Compute potential cost savings

The current tool kit supports:

- `get_weather_forecast`
- `get_electricity_prices`
- `query_energy_usage`
- `query_historical_energy_usage`
- `query_solar_generation`
- `analyze_solar_generation`
- `get_recent_energy_summary`
- `search_energy_tips`
- `calculate_energy_savings`

### Example Questions

The Energy Advisor can answer questions like:

- "When should I charge my electric car tomorrow to minimize cost and maximize solar power?"
- "What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?"
- "Suggest three ways I can reduce energy use based on my usage history."
- "How much can I save by running my dishwasher during off-peak hours?"

## Database Schema

### Energy Usage Table
- `timestamp`: When the energy was consumed
- `consumption_kwh`: Amount of energy used
- `device_type`: Type of device (EV, HVAC, appliance)
- `device_name`: Specific device name
- `cost_usd`: Cost at time of usage

### Solar Generation Table
- `timestamp`: When the energy was generated
- `generation_kwh`: Amount of solar energy produced
- `weather_condition`: Weather during generation
- `battery_storage_level`: Battery charge level recorded with solar output
- `exported_to_grid_kwh`: Excess solar exported back to the grid

## Evaluation Workflow

The evaluation notebook uses the in-project evaluation package under `evaluation/` and scores the agent on:

- Accuracy
- Relevance
- Completeness
- Usefulness
- Tool appropriateness
- Tool completeness

The notebook also generates a structured final report and a copyable submission log bundle.

## Submission Notes

- This project includes the database artifact, the vector store, expanded energy-saving documents, and the evaluation helpers inside the project folder.
- The README assumes the notebook and Python script outputs are part of the submission evidence.
- Include a screenshot of a successful local run as proof in the final submission package. Mention in the submission notes that local execution proof is provided as a screenshot.

## Learning Objectives

This project helps students learn:

1. **Database Design**: Creating schemas for energy management systems
2. **API Integration**: Working with external weather and pricing APIs
3. **RAG Implementation**: Building retrieval-augmented generation pipelines
4. **Agent Development**: Creating intelligent agents with tool usage
5. **Evaluation Methods**: Testing and measuring agent performance
6. **Energy Optimization**: Understanding smart home energy management

## Key Technologies

- **LangChain**: Agent framework and tool integration
- **LangGraph**: Agent orchestration and workflow
- **ChromaDB**: Vector database for document retrieval
- **SQLAlchemy**: Database ORM and management
- **OpenAI**: LLM and embeddings
- **SQLite**: Local database storage

## Evaluation Criteria

The agent is evaluated on:

- **Accuracy**: Correct information and calculations
- **Relevance**: Responses address the user's question
- **Completeness**: Comprehensive answers with actionable advice
- **Tool Usage**: Appropriate use of available tools
- **Reasoning**: Clear explanation of recommendations

## Getting Started

1. Install the required dependencies from [requirements.txt](c:/Users/BharathKumar/OneDrive%20-%20Synapx/lcgraph/EcoHome%20Project/requirements.txt)
2. Set up the environment variables above
3. Run the notebooks in sequence
4. Validate the agent with [03_run_and_evaluate.ipynb](c:/Users/BharathKumar/OneDrive%20-%20Synapx/lcgraph/EcoHome%20Project/03_run_and_evaluate.ipynb)
5. Test the agent with your own questions

## Contributing

This is a learning project. Feel free to:
- Add new tools and capabilities
- Improve the evaluation metrics
- Enhance the RAG pipeline
- Add more sophisticated optimization algorithms

## License

This project is for educational purposes as part of the Udacity Course 2 curriculum.
