"""AI Research & Weather Agent - Core Agent Engine.

This module initializes and manages a LangChain ReAct agent integrating:
- Groq LLM (qwen/qwen3.8-27b)
- Tavily Search API (web research)
- Weatherstack API (real-time weather data)
- SSL certificate handling for Windows environments
"""

import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import certifi
import requests
from dotenv import load_dotenv

# Suppress Pydantic V2 migration warnings and LangChain deprecation noise
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from pydantic.warnings import PydanticDeprecatedSince20
    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
except ImportError:
    pass

# Fix SSL certificate path for Windows environments
os.environ["SSL_CERT_FILE"] = certifi.where()

# Load environment variables (.env) from current or parent directory
CURRENT_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
ENV_PATH = CURRENT_DIR / ".env" if (CURRENT_DIR / ".env").exists() else CURRENT_DIR.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# LangChain imports
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq


# --- Credential Helpers ---

def get_api_credentials() -> Dict[str, str]:
    """Retrieve current API credentials from environment variables."""
    return {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", "").strip(),
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "").strip(),
        "WEATHERSTACK_API_KEY": os.getenv("WEATHERSTACK_API_KEY", "").strip(),
    }


def validate_credentials() -> Dict[str, bool]:
    """Check whether required API credentials are present and non-empty.

    Does not make network or API requests.

    Returns:
        Dict[str, bool]: A dictionary mapping credential names to booleans:
            - 'GROQ_API_KEY': True if set and non-empty, False otherwise.
            - 'TAVILY_API_KEY': True if set and non-empty, False otherwise.
            - 'WEATHERSTACK_API_KEY': True if set and non-empty, False otherwise.
    """
    creds = get_api_credentials()
    return {
        "GROQ_API_KEY": bool(creds["GROQ_API_KEY"]),
        "TAVILY_API_KEY": bool(creds["TAVILY_API_KEY"]),
        "WEATHERSTACK_API_KEY": bool(creds["WEATHERSTACK_API_KEY"]),
    }


def are_credentials_valid() -> bool:
    """Check if all required API credentials are valid and non-empty.

    Returns:
        bool: True if all three keys (GROQ, TAVILY, WEATHERSTACK) are present.
    """
    status = validate_credentials()
    return all(status.values())


# --- Custom Weather Tool ---

@tool
def get_weather(city: str) -> str:
    """Fetch current weather information for a specific city.

    Input should be the name of a city (e.g., 'Karachi', 'London', 'Tokyo').
    Returns structured temperature, weather condition, and humidity.
    """
    clean_city = str(city).strip().strip("\"'").strip()
    if not clean_city:
        return "Error: City name cannot be empty."

    api_key = os.getenv("WEATHERSTACK_API_KEY", "").strip()
    if not api_key:
        return "Error: WEATHERSTACK_API_KEY is not configured in the environment or .env file."

    # Weatherstack free tier accounts only support HTTP. Paid accounts support HTTPS.
    # We attempt HTTPS first; if restricted (code 105) or SSL fails, we fallback to HTTP.
    endpoints = [
        "https://api.weatherstack.com/current",
        "http://api.weatherstack.com/current",
    ]

    last_error = "Unknown error"
    for url in endpoints:
        try:
            params = {
                "access_key": api_key,
                "query": clean_city,
            }
            response = requests.get(url, params=params, timeout=12)
            data = response.json()

            # Check for API error response
            if "error" in data:
                err_info = data.get("error", {})
                err_code = err_info.get("code")
                err_msg = err_info.get("info", "API error")
                # Code 105: HTTPS access restricted on free plan -> try HTTP fallback
                if err_code == 105 and url.startswith("https://"):
                    continue
                return f"Could not fetch weather data for {clean_city}. Reason: {err_msg}"

            if "current" not in data:
                return f"Could not fetch weather data for {clean_city}. Incomplete response from Weatherstack."

            current = data["current"]
            temp = current.get("temperature", "N/A")
            descriptions = current.get("weather_descriptions", ["Unknown"])
            description = descriptions[0] if descriptions else "Unknown"
            humidity = current.get("humidity", "N/A")
            location_info = data.get("location", {})
            resolved_city = location_info.get("name", clean_city)

            return (
                f"City: {resolved_city}\n"
                f"Temperature: {temp}°C\n"
                f"Weather: {description}\n"
                f"Humidity: {humidity}%"
            )
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            continue
        except Exception as e:
            return f"Error retrieving weather for {clean_city}: {str(e)}"

    return f"Error retrieving weather for {clean_city}: {last_error}"


# --- Weather Extraction Helper ---

def parse_weather_reports(text: str) -> List[Dict[str, str]]:
    """Parse structured weather reports from text or tool observations.

    Looks for patterns formatted like:
        City: <name>
        Temperature: <num>°C
        Weather: <condition>
        Humidity: <num>%
    """
    reports: List[Dict[str, str]] = []
    pattern = re.compile(
        r"City:\s*([^\n\r]+)[\r\n]+"
        r"Temperature:\s*([^\n\r°]+)(?:°C)?[\r\n]+"
        r"Weather:\s*([^\n\r]+)[\r\n]+"
        r"Humidity:\s*([^\n\r%]+)%?",
        re.IGNORECASE,
    )

    for match in pattern.finditer(text):
        city = match.group(1).strip()
        temp = match.group(2).strip()
        weather = match.group(3).strip()
        humidity = match.group(4).strip()
        reports.append({
            "city": city,
            "temperature": temp,
            "weather": weather,
            "humidity": humidity,
        })

    return reports


# --- Agent Factory and Execution ---

# Standard ReAct prompt fallback in case LangChain Hub is temporarily unreachable
REACT_PROMPT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def get_agent_executor(
    temperature: float = 0.0,
    max_tokens: int = 600,
    model_name: str = "qwen/qwen3.8-27b",
    return_intermediate_steps: bool = True,
) -> AgentExecutor:
    """Build and return a LangChain ReAct AgentExecutor.

    Args:
        temperature: Creativity setting for Groq LLM (0.0 to 1.0).
        max_tokens: Maximum tokens in LLM generation.
        model_name: Groq model identifier (default: qwen/qwen3.8-27b).
        return_intermediate_steps: Whether to return tool invocation steps.

    Raises:
        ValueError: If required API keys are missing.
    """
    creds = get_api_credentials()
    missing = [k for k, v in creds.items() if not v]
    if missing:
        raise ValueError(
            f"Missing required API credentials: {', '.join(missing)}. "
            f"Please configure them in your environment or .env file."
        )

    # 1. Initialize Tavily Search Tool
    search_tool = TavilySearchResults(
        max_results=3,
        tavily_api_key=creds["TAVILY_API_KEY"],
    )

    # 2. Tools list
    tools = [search_tool, get_weather]

    # 3. Initialize Groq LLM
    llm = ChatGroq(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=creds["GROQ_API_KEY"],
    )

    # 4. Pull ReAct prompt from LangChain Hub (with reliable local fallback)
    try:
        prompt = hub.pull("hwchase17/react")
    except Exception:
        prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

    # 5. Build ReAct Agent
    agent_obj = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    # 6. Build Agent Executor
    executor = AgentExecutor(
        agent=agent_obj,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=8,
    )

    return executor


def execute_agent_query(
    user_query: str,
    executor: Optional[AgentExecutor] = None,
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> Dict[str, Any]:
    """Execute a user query through the agent and return formatted results.

    Returns:
        Dict containing:
            - 'output': final answer string
            - 'intermediate_steps': list of (AgentAction, tool_output) tuples
            - 'weather_reports': list of parsed weather dicts
    """
    if not executor:
        executor = get_agent_executor(temperature=temperature, max_tokens=max_tokens)

    response = executor.invoke({"input": user_query})
    output = response.get("output", "")
    intermediate_steps = response.get("intermediate_steps", [])

    # Extract structured weather reports from tool observations first
    weather_reports: List[Dict[str, str]] = []
    seen_cities = set()

    for action, observation in intermediate_steps:
        if getattr(action, "tool", None) == "get_weather" or "City:" in str(observation):
            parsed = parse_weather_reports(str(observation))
            for r in parsed:
                city_key = r["city"].lower()
                if city_key not in seen_cities:
                    seen_cities.add(city_key)
                    weather_reports.append(r)

    # Fallback: check final text output if no tool observation matched
    if not weather_reports and "City:" in output:
        weather_reports = parse_weather_reports(output)

    return {
        "output": output,
        "intermediate_steps": intermediate_steps,
        "weather_reports": weather_reports,
    }


def main():
    """CLI Entry point for direct script execution."""
    print("=" * 60)
    print("AI Research & Weather Agent - CLI Interface")
    print("=" * 60)

    # Validate credentials
    status = validate_credentials()
    missing = [k for k, v in status.items() if not v]
    if missing:
        sys.exit(f"Error: Missing required API keys in .env: {', '.join(missing)}")

    print("Initializing components...")
    executor = get_agent_executor(temperature=0.0, max_tokens=600)

    # Query from command line argument or default
    query = sys.argv[1] if len(sys.argv) > 1 else "Find the capital of Pakistan and its current weather?"
    print(f"\nRunning Query: {query}\n")

    result = execute_agent_query(query, executor=executor)

    print("\n" + "=" * 50)
    print("FINAL ANSWER:")
    print("=" * 50)
    print(result.get("output"))

    weather_reports = result.get("weather_reports", [])
    if weather_reports:
        print("\nStructured Weather Data Detected:")
        for r in weather_reports:
            print(f" -> {r['city']}: {r['temperature']}°C, {r['weather']}, Humidity: {r['humidity']}%")


if __name__ == "__main__":
    main()