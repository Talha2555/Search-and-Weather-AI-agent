# ⚡ AI Research & Weather Agent

An enterprise-ready AI Agent web application built with **Streamlit** and **LangChain ReAct**, powered by **Groq (`qwen/qwen3.8-27b`)**, **Tavily Search**, and **Weatherstack**.

---

## 🌟 Overview

The **AI Research & Weather Agent** receives natural language questions and autonomously decides whether to perform real-time web searches, query live meteorological data, or synthesize information directly using a ReAct (Reasoning + Acting) loop.

### Key Capabilities:
- 🧠 **Autonomous Decision Making**: Uses the ReAct pattern to decide which tools to use and how to combine results.
- 🌐 **Web Intelligence**: Real-time web search powered by the Tavily Search API.
- 🌦️ **Live Weather Telemetry**: Direct integration with Weatherstack to fetch real-time temperatures, conditions, and humidity with structured card presentation.
- 💬 **SaaS Dashboard Chat Interface**: Built using native Streamlit chat components (`st.chat_message`, `st.chat_input`) with conversation export and state management.
- 🛡️ **Enterprise Security & Reliability**: Safe environment variable management, graceful error handling (handling rate limits and API failures without crashing), and Windows SSL certificate handling via `certifi`.

---

## 🏗️ Architecture

```
User Query (Streamlit UI: app.py)
               │
               ▼
    LangChain ReAct Agent
    (Groq: qwen/qwen3.8-27b)
       │              │
       ▼              ▼
 Tavily Search    Weatherstack
   (Web Data)     (Live Weather)
       │              │
       └──────┬───────┘
              ▼
   Structured Output & Trace
(Metrics, Markdown & Activity)
```

---

## 📋 Prerequisites

- **Python 3.10 to 3.13** installed on your system.
- Free API keys from:
  - [Groq Cloud](https://console.groq.com/keys) (Ultra-fast LLM inference)
  - [Tavily Search](https://app.tavily.com) (Agentic web search)
  - [Weatherstack](https://weatherstack.com) (Real-time weather data)

---

## 🚀 Quickstart Guide

### 1. Clone or Open the Repository
```bash
git clone <your-repo-url>
cd langchain_react_agent
```

### 2. Set Up a Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### If using Conda:
```bash
conda create -n langagent python=3.11 -y
conda activate langagent
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the template `.env.example` to create your `.env` file:

```bash
# On Windows PowerShell:
Copy-Item .env.example .env

# On macOS / Linux:
cp .env.example .env
```

Open `.env` and fill in your actual credentials:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
TAVILY_API_KEY=tvly-your_tavily_api_key_here
WEATHERSTACK_API_KEY=your_weatherstack_api_key_here
```

> **Security Note:** `.env` is listed in `.gitignore` to prevent committing secrets to version control.

---

## 🖥️ Running the Application

### Launch Streamlit Web UI:
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

### (Optional) Run CLI Mode:
You can also execute individual queries directly from your terminal:
```bash
python agent.py "Find the capital of Pakistan and its current weather"
```

---

## 📁 Project Structure

```
langchain_react_agent/
├── agent.py            # LangChain ReAct agent core, tools, & execution logic
├── app.py              # Streamlit web application & modern SaaS UI
├── requirements.txt    # Project dependencies
├── .env.example        # Safe template for required environment variables
├── .env                # Local secrets (never committed)
├── .gitignore          # Git exclusion rules for secrets and caches
└── README.md           # Documentation and setup instructions
```

---

## 🛠️ Features & Controls

| Feature | Description |
| :--- | :--- |
| **Interactive Examples** | Clickable queries in the sidebar to test research, weather, and comparisons immediately. |
| **Model Controls** | Adjust response temperature (0.0 - 1.0) and maximum response tokens directly from the sidebar. |
| **Structured Weather Cards** | Automatic parsing of temperatures, conditions, and humidity into multi-city metric cards. |
| **Execution Trace Expander** | Collapsible reasoning drawer displaying genuine tool calls and inputs without exposing API keys. |
| **Conversation Management** | Clear chat session state or export full conversation transcripts as Markdown. |

---

## 🔍 Troubleshooting

### 1. Windows SSL Certificate Error
If you encounter `SSLCertVerificationError` on Windows, `agent.py` automatically binds `certifi` via:
```python
os.environ["SSL_CERT_FILE"] = certifi.where()
```
Ensure `certifi` is installed via `pip install certifi`.

### 2. Groq Rate Limits (`OTPM` / `TPM`)
If you encounter Groq token rate limits, lower the **Max Response Tokens** slider in the sidebar (e.g. to 400 or 500) and ensure `temperature` is set to 0.

### 3. Weatherstack Free Tier (HTTPS Restriction)
Weatherstack's free plan uses HTTP (`http://api.weatherstack.com/current`). `agent.py` automatically detects free plan limitations (code 105) and falls back from HTTPS to HTTP transparently.

### 4. Streamlit Command Not Found
If running `streamlit run app.py` says `streamlit: command not found`, run:
```bash
python -m streamlit run app.py
```
Ensure your virtual environment or conda environment is active.
