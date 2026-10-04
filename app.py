"""AI Research & Weather Agent - Streamlit Application.

A modern, production-grade interface for LangChain ReAct agent integrating:
- Groq LLM (Qwen 27B)
- Tavily Web Search API
- Weatherstack Real-time Weather API
"""

import html
import json
import os
from typing import Any, Dict, List, Optional

import streamlit as st

# Import backend agent logic
import agent

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom SaaS Styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Research & Weather Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Refined SaaS CSS for a clean, modern dashboard aesthetic
st.markdown(
    """
    <style>
    /* Global Font & Layout Adjustments */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1050px;
    }

    /* Header Container Styling */
    .agent-header {
        padding: 1.2rem 1.5rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.05) 0%, rgba(51, 65, 85, 0.02) 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .agent-title-group h1 {
        font-size: 1.65rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
        color: inherit;
    }

    .agent-title-group p {
        font-size: 0.92rem;
        margin: 0.3rem 0 0 0;
        opacity: 0.8;
    }

    /* Status Pill Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.4rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    .status-ready {
        background-color: rgba(34, 197, 94, 0.12);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .status-warning {
        background-color: rgba(234, 179, 8, 0.12);
        color: #ca8a04;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: currentColor;
    }

    /* Welcome Hero Card */
    .welcome-card {
        padding: 2rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        background: rgba(248, 250, 252, 0.4);
        margin-bottom: 1.5rem;
    }

    .welcome-card h3 {
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }

    .capability-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        background: rgba(148, 163, 184, 0.12);
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }

    /* Weather Metric Card Styling */
    .weather-card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .weather-card {
        flex: 1 1 200px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.06) 0%, rgba(99, 102, 241, 0.04) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }

    .weather-card-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.5rem;
    }

    .weather-city {
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0;
    }

    .weather-temp {
        font-size: 1.7rem;
        font-weight: 700;
        color: #2563eb;
        margin: 0;
    }

    .weather-meta {
        font-size: 0.85rem;
        opacity: 0.85;
        display: flex;
        justify-content: space-between;
        margin-top: 0.4rem;
        padding-top: 0.4rem;
        border-top: 1px dashed rgba(148, 163, 184, 0.25);
    }

    /* Activity Step Card */
    .tool-step-box {
        font-size: 0.85rem;
        padding: 0.65rem 0.9rem;
        border-left: 3px solid #3b82f6;
        background: rgba(148, 163, 184, 0.08);
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }

    /* Streamlit Chat tweaks */
    [data-testid="stChatMessage"] {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }

    /* Sidebar polish */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }
    
    .sidebar-section-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        opacity: 0.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# -----------------------------------------------------------------------------
# 3. Agent Caching & Management
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_cached_executor(temperature: float, max_tokens: int) -> Optional[Any]:
    """Initialize and cache the LangChain AgentExecutor.

    Reinitializes automatically when temperature or max_tokens parameters change.
    Returns None if required API credentials are missing.
    """
    if not agent.are_credentials_valid():
        return None
    try:
        return agent.get_agent_executor(
            temperature=temperature,
            max_tokens=max_tokens,
            return_intermediate_steps=True,
        )
    except Exception as exc:
        st.session_state["init_error"] = str(exc)
        return None


# -----------------------------------------------------------------------------
# 4. Sidebar: Settings, Capabilities, and Example Queries
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚡ Agent Control Hub")
    st.caption("Orchestrating Groq LLM, Tavily Search, & Weatherstack")

    st.markdown("---")

    # A. Agent Settings Section
    st.markdown('<div class="sidebar-section-title">Model Settings</div>', unsafe_allow_html=True)
    temperature = st.slider(
        "Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="0.0 provides deterministic, fact-focused responses. Higher values increase creativity.",
    )

    max_tokens = st.slider(
        "Max Response Tokens",
        min_value=200,
        max_value=1500,
        value=600,
        step=50,
        help="Controls the maximum token generation limit to avoid OTPM rate limits on Groq.",
    )

    if temperature > 0.0 or max_tokens != 600:
        st.caption("ℹ️ *Adjusted settings apply to all subsequent queries.*")

    st.markdown("---")

    # B. Example Queries Section
    st.markdown('<div class="sidebar-section-title">Example Queries</div>', unsafe_allow_html=True)
    st.caption("Click any prompt to run it through the agent:")

    examples = [
        ("🏛️ Capital of Pakistan", "What is the capital of Pakistan?"),
        ("🌦️ Karachi Weather", "What is the current weather in Karachi?"),
        ("⚖️ Weather Comparison", "Compare the weather in Karachi and Lahore."),
        ("🔬 AI Research", "Research the latest developments in artificial intelligence."),
        ("🌐 Combined Query", "Find the capital of Pakistan and its current weather."),
    ]

    for label, query_text in examples:
        if st.button(label, use_container_width=True, key=f"ex_{label}"):
            st.session_state.pending_query = query_text

    st.markdown("---")

    # C. System & Credential Status
    st.markdown('<div class="sidebar-section-title">Environment & Integrations</div>', unsafe_allow_html=True)
    cred_status = agent.validate_credentials()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(
            f"**Groq LLM:** {'✅' if cred_status['GROQ_API_KEY'] else '❌'}<br>"
            f"<small>{'Connected' if cred_status['GROQ_API_KEY'] else 'Missing'}</small>",
            unsafe_allow_html=True,
        )
    with col_s2:
        st.markdown(
            f"**Tavily Search:** {'✅' if cred_status['TAVILY_API_KEY'] else '❌'}<br>"
            f"<small>{'Connected' if cred_status['TAVILY_API_KEY'] else 'Missing'}</small>",
            unsafe_allow_html=True,
        )

    col_s3, col_s4 = st.columns(2)
    with col_s3:
        st.markdown(
            f"**Weatherstack:** {'✅' if cred_status['WEATHERSTACK_API_KEY'] else '❌'}<br>"
            f"<small>{'Connected' if cred_status['WEATHERSTACK_API_KEY'] else 'Missing'}</small>",
            unsafe_allow_html=True,
        )
    with col_s4:
        st.markdown(
            "**Windows SSL:** 🛡️<br><small>Certifi Active</small>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # D. Conversation Controls
    st.markdown('<div class="sidebar-section-title">Conversation Controls</div>', unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()

    with col_c2:
        if st.session_state.messages:
            # Prepare conversation history export text
            export_lines = ["# AI Research & Weather Agent - Conversation Log\n"]
            for msg in st.session_state.messages:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                export_lines.append(f"### {role_label}:\n{msg['content']}\n")
            export_text = "\n".join(export_lines)
            st.download_button(
                "📥 Export",
                data=export_text,
                file_name="agent_conversation.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.button("📥 Export", disabled=True, use_container_width=True)

    # E. About Section
    with st.expander("ℹ️ About the Architecture"):
        st.markdown(
            """
            **Framework:** LangChain ReAct (`AgentExecutor`)
            
            **LLM Engine:** Groq Cloud (`qwen/qwen3.8-27b`)
            
            **Autonomous Decision Making:** The agent analyzes the user's natural language input and dynamically decides whether to execute a Web Search via Tavily, fetch live weather via Weatherstack, or synthesize knowledge directly.
            
            *Note: Each query is executed autonomously through the ReAct loop.*
            """
        )


# -----------------------------------------------------------------------------
# 5. Main Header & Readiness Check
# -----------------------------------------------------------------------------

all_configured = agent.are_credentials_valid()

st.markdown(
    f"""
    <div class="agent-header">
        <div class="agent-title-group">
            <h1>⚡ AI Research & Weather Agent</h1>
            <p>An intelligent assistant for web research and real-time weather information.</p>
        </div>
        <div>
            {
                '<div class="status-badge status-ready"><div class="status-dot"></div> System Ready</div>'
                if all_configured
                else '<div class="status-badge status-warning"><div class="status-dot"></div> Configuration Required</div>'
            }
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display friendly configuration instructions if any credentials are missing
if not all_configured:
    missing_keys = [k for k, v in cred_status.items() if not v]
    st.warning(
        f"⚠️ **Missing API Keys:** {', '.join(missing_keys)}\n\n"
        "To enable real-time research and weather lookups, provide your credentials in a `.env` file in the project directory:\n\n"
        "```bash\n"
        "GROQ_API_KEY=your_groq_key\n"
        "TAVILY_API_KEY=your_tavily_key\n"
        "WEATHERSTACK_API_KEY=your_weatherstack_key\n"
        "```\n"
        "See `.env.example` for details. You can get keys for free at: "
        "[Groq Console](https://console.groq.com), "
        "[Tavily Search](https://app.tavily.com), and "
        "[Weatherstack](https://weatherstack.com)."
    )


# -----------------------------------------------------------------------------
# 6. Weather Presentation Helper
# -----------------------------------------------------------------------------

def render_weather_metrics(weather_reports: List[Dict[str, str]]) -> None:
    """Render structured weather reports in a polished multi-column layout."""
    if not weather_reports:
        return

    st.markdown("##### 🌦️ Weather Report Summary")
    cols = st.columns(min(len(weather_reports), 3))

    for idx, report in enumerate(weather_reports):
        col = cols[idx % len(cols)]
        with col:
            city = report.get("city", "Unknown City")
            temp = report.get("temperature", "--")
            cond = report.get("weather", "Unknown")
            hum = report.get("humidity", "--")

            st.markdown(
                f"""
                <div class="weather-card">
                    <div class="weather-card-header">
                        <span class="weather-city">📍 {html.escape(city)}</span>
                    </div>
                    <div class="weather-temp">{html.escape(str(temp))}°C</div>
                    <div class="weather-meta">
                        <span>Condition: <strong>{html.escape(cond)}</strong></span>
                        <span>Humidity: <strong>{html.escape(str(hum))}%</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.write("")  # Margin spacing


# -----------------------------------------------------------------------------
# 7. Agent Activity (Execution Trace) Presentation Helper
# -----------------------------------------------------------------------------

def render_agent_activity(intermediate_steps: List[Any]) -> None:
    """Render genuine tool calls and intermediate thoughts captured during execution."""
    if not intermediate_steps:
        return

    with st.expander("🔍 View Agent Reasoning & Tool Activity", expanded=False):
        for idx, step in enumerate(intermediate_steps, 1):
            if isinstance(step, tuple) and len(step) == 2:
                action, observation = step
                tool_name = getattr(action, "tool", "Unknown Tool")
                tool_input = getattr(action, "tool_input", "")
                thought_log = getattr(action, "log", "")

                st.markdown(f"**Step {idx}: Executed Tool `{tool_name}`**")

                if thought_log:
                    # Filter out raw prompt leaks, keep concise thought
                    clean_thought = thought_log.split("Action:")[0].replace("Thought:", "").strip()
                    if clean_thought:
                        st.caption(f"🧠 *Reasoning:* {clean_thought}")

                st.code(f"Tool: {tool_name}\nInput: {tool_input}", language="text")

                # Sanitize observation string for display
                obs_str = str(observation).strip()
                if len(obs_str) > 350:
                    obs_str = obs_str[:350] + " ... [Output truncated for readability]"

                st.markdown(
                    f'<div class="tool-step-box"><strong>Observation:</strong><br>{html.escape(obs_str)}</div>',
                    unsafe_allow_html=True,
                )


# -----------------------------------------------------------------------------
# 8. Render Chat Messages
# -----------------------------------------------------------------------------

# If no messages yet, display an elegant welcome card
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>👋 Welcome to AI Research & Weather Agent</h3>
            <p>
                This agent combines fast LLM reasoning with real-time web intelligence. Ask queries that require 
                live web searches, current weather conditions across cities, or multi-step reasoning.
            </p>
            <div style="margin-top: 1rem;">
                <span class="capability-tag">⚡ Groq Qwen 27B</span>
                <span class="capability-tag">🔍 Tavily Web Search</span>
                <span class="capability-tag">🌦️ Real-Time Weatherstack</span>
                <span class="capability-tag">🛡️ SSL Hardened</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render previous messages from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Display structured weather summary if present
            if msg.get("weather_reports"):
                render_weather_metrics(msg["weather_reports"])

            # Render textual Markdown answer
            st.markdown(msg["content"])

            # Display real agent execution trace if recorded
            if msg.get("intermediate_steps"):
                render_agent_activity(msg["intermediate_steps"])
        else:
            st.markdown(msg["content"])


# -----------------------------------------------------------------------------
# 9. Query Handling and Execution Flow
# -----------------------------------------------------------------------------

# Check for input from either chat_input or sidebar example button click
chat_input_query = st.chat_input("Ask a question about current research or weather...")
query_to_process = None

if st.session_state.pending_query:
    query_to_process = st.session_state.pending_query
    st.session_state.pending_query = None
elif chat_input_query and chat_input_query.strip():
    query_to_process = chat_input_query.strip()

if query_to_process:
    # 1. Display user query immediately
    st.session_state.messages.append({"role": "user", "content": query_to_process})
    with st.chat_message("user"):
        st.markdown(query_to_process)

    # 2. Check credentials readiness before invoking agent
    if not all_configured:
        err_msg = (
            "Cannot execute query: One or more required API credentials are missing. "
            "Please check your `.env` configuration file."
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ {err_msg}",
            "weather_reports": [],
            "intermediate_steps": [],
        })
        with st.chat_message("assistant"):
            st.error(err_msg)
    else:
        # 3. Initialize agent executor and run query
        with st.chat_message("assistant"):
            with st.spinner("Researching your query and retrieving live data..."):
                try:
                    executor = get_cached_executor(temperature=temperature, max_tokens=max_tokens)
                    if executor is None:
                        raise RuntimeError("Agent executor could not be initialized.")

                    result = agent.execute_agent_query(
                        user_query=query_to_process,
                        executor=executor,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    answer_text = result.get("output", "No response generated.")
                    weather_data = result.get("weather_reports", [])
                    tool_steps = result.get("intermediate_steps", [])

                    # Render structured weather if found
                    if weather_data:
                        render_weather_metrics(weather_data)

                    # Render final answer text
                    st.markdown(answer_text)

                    # Render agent activity expander
                    if tool_steps:
                        render_agent_activity(tool_steps)

                    # Save to session history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "weather_reports": weather_data,
                        "intermediate_steps": tool_steps,
                    })

                except Exception as exc:
                    error_description = str(exc)
                    # Friendly masking of technical details or API keys
                    if "rate limit" in error_description.lower():
                        user_friendly_error = (
                            "Groq API rate limit reached (tokens/requests per minute). "
                            "Please wait a moment before trying again, or reduce the Max Response Tokens setting."
                        )
                    elif "timeout" in error_description.lower():
                        user_friendly_error = (
                            "Network request timed out while contacting search or weather services. "
                            "Please check your internet connection."
                        )
                    else:
                        user_friendly_error = f"Agent encountered an execution error: {error_description}"

                    st.error(user_friendly_error)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ {user_friendly_error}",
                        "weather_reports": [],
                        "intermediate_steps": [],
                    })
