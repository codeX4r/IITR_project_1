import io
import json
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import streamlit as st
import yaml
from crewai import Agent, Crew, LLM, Process, Task


# =========================================================
# Streamlit Config
# =========================================================

st.set_page_config(
    page_title="CrewAI Delegation Chat",
    page_icon="🤖",
    layout="wide",
)


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
AGENTS_YAML_PATH = CONFIG_DIR / "agents.yaml"
TASKS_YAML_PATH = CONFIG_DIR / "tasks.yaml"


# =========================================================
# Defaults
# =========================================================

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL_NAME = "llama3.2:3b"


# =========================================================
# Event Helpers
# =========================================================

def now_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


def add_event(
    events: List[Dict[str, Any]],
    event_type: str,
    title: str,
    detail: str = "",
    agent: str = "System",
) -> None:
    events.append(
        {
            "time": now_time(),
            "type": event_type,
            "agent": agent,
            "title": title,
            "detail": detail,
        }
    )


def event_icon(event_type: str) -> str:
    icons = {
        "thinking": "💭",
        "reasoning": "🧠",
        "delegating": "🔁",
        "executing": "⚙️",
        "completed": "✅",
        "failed": "❌",
        "info": "ℹ️",
        "callback": "📍",
    }
    return icons.get(event_type, "•")


def render_event_timeline(events: List[Dict[str, Any]], placeholder) -> None:
    """
    Render a live activity timeline into a replaceable placeholder.

    IMPORTANT:
    Use this with `st.empty()`. If you use `st.container()` and keep
    writing to it repeatedly, Streamlit will append duplicate timelines.
    """
    placeholder.empty()

    with placeholder.container():
        st.markdown("#### Live Agent Activity")

        if not events:
            st.caption("No events yet.")
            return

        for event in events[-30:]:
            icon = event_icon(event.get("type", "info"))
            time_value = event.get("time", "")
            agent = event.get("agent", "System")
            title = event.get("title", "")
            detail = event.get("detail", "")

            st.markdown(
                f"""
                <div style="
                    padding: 10px 12px;
                    margin-bottom: 8px;
                    border-radius: 10px;
                    border: 1px solid rgba(128,128,128,0.25);
                ">
                    <b>{icon} {title}</b><br/>
                    <span style="font-size: 0.85rem; opacity: 0.75;">
                        {time_value} · {agent}
                    </span><br/>
                    <span>{detail}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )



# =========================================================
# YAML Helpers
# =========================================================

@st.cache_data
def load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    agents_config = load_yaml_file(AGENTS_YAML_PATH)
    tasks_config = load_yaml_file(TASKS_YAML_PATH)
    return agents_config, tasks_config


# =========================================================
# Ollama Helpers
# =========================================================

def get_ollama_models(base_url: str) -> List[str]:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()

        data = response.json()
        models = [model.get("name") for model in data.get("models", [])]

        return [model for model in models if model]

    except Exception:
        return []


def build_llm(
    model_name: str,
    base_url: str,
    temperature: float,
) -> LLM:
    return LLM(
        model=f"ollama/{model_name}",
        base_url=base_url,
        temperature=temperature,
    )


# =========================================================
# Metrics Helpers
# =========================================================

def estimate_tokens(text: str) -> int:
    """
    Approx token estimation.

    Exact token usage can vary with Ollama + LiteLLM + CrewAI.
    This gives practical context-window visibility.

    Rule:
    1 token ~= 4 characters.
    """
    if not text:
        return 0

    return max(1, len(text) // 4)


def safe_json(obj: Any) -> Dict[str, Any]:
    try:
        if obj is None:
            return {}

        if isinstance(obj, dict):
            return obj

        if hasattr(obj, "model_dump"):
            return obj.model_dump()

        if hasattr(obj, "dict"):
            return obj.dict()

        if hasattr(obj, "__dict__"):
            return vars(obj)

        return {"value": str(obj)}

    except Exception:
        return {"value": str(obj)}


def build_chat_history(
    messages: List[Dict[str, Any]],
    max_messages: int,
) -> str:
    recent_messages = messages[-max_messages:]
    lines = []

    for msg in recent_messages:
        role = msg.get("role", "unknown").upper()
        agent_name = msg.get("agent_name", "")
        content = msg.get("content", "")

        if role == "ASSISTANT" and agent_name:
            lines.append(f"{role} ({agent_name}): {content}")
        else:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


def calculate_context_metrics(
    chat_history: str,
    user_prompt: str,
    agents_config: Dict[str, Any],
    tasks_config: Dict[str, Any],
    context_window_tokens: int,
) -> Dict[str, Any]:
    static_config_text = json.dumps(
        {
            "agents": agents_config,
            "tasks": tasks_config,
        },
        ensure_ascii=False,
    )

    chat_history_tokens = estimate_tokens(chat_history)
    user_prompt_tokens = estimate_tokens(user_prompt)
    static_config_tokens = estimate_tokens(static_config_text)

    estimated_total_input_tokens = (
        chat_history_tokens
        + user_prompt_tokens
        + static_config_tokens
    )

    estimated_context_usage_percent = round(
        (estimated_total_input_tokens / context_window_tokens) * 100,
        2,
    )

    return {
        "chat_history_tokens": chat_history_tokens,
        "user_prompt_tokens": user_prompt_tokens,
        "static_config_tokens": static_config_tokens,
        "estimated_total_input_tokens": estimated_total_input_tokens,
        "context_window_tokens": context_window_tokens,
        "estimated_context_usage_percent": estimated_context_usage_percent,
    }


# =========================================================
# CrewAI Callback Helpers
# =========================================================

def make_step_callback(
    events: List[Dict[str, Any]],
    timeline_placeholder,
) -> Callable[[Any], None]:
    """
    CrewAI step callback.

    CrewAI callback payload shape can vary by version and event type.
    This callback is defensive and only extracts safe display text.
    """

    def step_callback(step_output: Any) -> None:
        try:
            raw_text = str(step_output)
            short_text = raw_text[:700]

            lowered = raw_text.lower()

            if "delegate" in lowered or "coworker" in lowered:
                event_type = "delegating"
                title = "Delegation event detected"
            elif "thought" in lowered or "reasoning" in lowered or "think" in lowered:
                event_type = "reasoning"
                title = "Reasoning step captured"
            elif "tool" in lowered or "action" in lowered:
                event_type = "executing"
                title = "Execution step captured"
            else:
                event_type = "callback"
                title = "CrewAI step callback"

            add_event(
                events=events,
                event_type=event_type,
                title=title,
                detail=short_text,
                agent="CrewAI",
            )

            timeline_placeholder.empty()
            render_event_timeline(events, timeline_placeholder)

        except Exception:
            pass

    return step_callback


# =========================================================
# CrewAI Builders
# =========================================================

def build_agent_from_yaml(
    agent_key: str,
    agents_config: Dict[str, Any],
    llm: LLM,
    step_callback: Optional[Callable[[Any], None]] = None,
) -> Agent:
    cfg = agents_config[agent_key]

    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=llm,
        verbose=bool(cfg.get("verbose", True)),
        allow_delegation=bool(cfg.get("allow_delegation", False)),
        max_iter=int(cfg.get("max_iter", 5)),
        max_retry_limit=int(cfg.get("max_retry_limit", 2)),
        respect_context_window=True,
        use_system_prompt=True,
        step_callback=step_callback,
    )


def build_manager_task(
    tasks_config: Dict[str, Any],
    chat_history: str,
    user_prompt: str,
) -> Task:
    cfg = tasks_config["analytics_manager_task"]

    return Task(
        description=cfg["description"].format(
            chat_history=chat_history,
            user_prompt=user_prompt,
        ),
        expected_output=cfg["expected_output"],
    )


def extract_crew_result_text(result: Any) -> str:
    """
    Extract the best final answer text from CrewAI kickoff output.

    Depending on CrewAI version, kickoff() can return a CrewOutput object,
    a plain string, or an object with raw/final/task output fields.
    """
    if result is None:
        return ""

    for attr in ("raw", "final_output", "output"):
        value = getattr(result, attr, None)
        if value:
            return str(value)

    tasks_output = getattr(result, "tasks_output", None)
    if tasks_output:
        last_task = tasks_output[-1]
        for attr in ("raw", "description", "summary", "output"):
            value = getattr(last_task, attr, None)
            if value:
                return str(value)

    return str(result)


def run_delegation_crew(
    user_prompt: str,
    chat_history: str,
    agents_config: Dict[str, Any],
    tasks_config: Dict[str, Any],
    model_name: str,
    base_url: str,
    temperature: float,
    events: List[Dict[str, Any]],
    timeline_placeholder,
) -> Tuple[str, Dict[str, Any], str]:
    """
    Runs CrewAI native delegation using Process.hierarchical.

    Supervisor Agent is the manager_agent.
    Data Scientist and Data Analyst are worker agents.
    CrewAI decides delegation internally.
    """

    add_event(
        events,
        "thinking",
        "Building local Ollama LLM",
        f"Model: ollama/{model_name}",
        "System",
    )
    timeline_placeholder.empty()
    render_event_timeline(events, timeline_placeholder)

    llm = build_llm(
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
    )

    step_callback = make_step_callback(events, timeline_placeholder)

    add_event(
        events,
        "executing",
        "Creating CrewAI agents",
        "Supervisor, Data Scientist, and Data Analyst agents are being initialized.",
        "System",
    )
    timeline_placeholder.empty()
    render_event_timeline(events, timeline_placeholder)

    supervisor_agent = build_agent_from_yaml(
        agent_key="supervisor_agent",
        agents_config=agents_config,
        llm=llm,
        step_callback=step_callback,
    )

    data_scientist_agent = build_agent_from_yaml(
        agent_key="data_scientist_agent",
        agents_config=agents_config,
        llm=llm,
        step_callback=step_callback,
    )

    data_analyst_agent = build_agent_from_yaml(
        agent_key="data_analyst_agent",
        agents_config=agents_config,
        llm=llm,
        step_callback=step_callback,
    )

    add_event(
        events,
        "reasoning",
        "Preparing manager task",
        "Supervisor will inspect chat history and decide whether to delegate to one or both specialists.",
        "Supervisor Agent",
    )
    timeline_placeholder.empty()
    render_event_timeline(events, timeline_placeholder)

    manager_task = build_manager_task(
        tasks_config=tasks_config,
        chat_history=chat_history,
        user_prompt=user_prompt,
    )

    add_event(
        events,
        "delegating",
        "Starting hierarchical delegation",
        "CrewAI manager agent can delegate internally to specialist agents.",
        "Supervisor Agent",
    )
    timeline_placeholder.empty()
    render_event_timeline(events, timeline_placeholder)

    crew = Crew(
        agents=[
            data_scientist_agent,
            data_analyst_agent,
        ],
        tasks=[
            manager_task,
        ],
        manager_agent=supervisor_agent,
        process=Process.hierarchical,
        verbose=True,
    )

    trace_buffer = io.StringIO()

    with redirect_stdout(trace_buffer), redirect_stderr(trace_buffer):
        result = crew.kickoff()

    delegation_trace = trace_buffer.getvalue()
    usage_metrics = safe_json(getattr(crew, "usage_metrics", None))

    add_event(
        events,
        "completed",
        "CrewAI run completed",
        "Supervisor has returned the final response.",
        "Supervisor Agent",
    )
    timeline_placeholder.empty()
    render_event_timeline(events, timeline_placeholder)

    final_response = extract_crew_result_text(result)

    return final_response, usage_metrics, delegation_trace


# =========================================================
# Session State
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_usage_metrics" not in st.session_state:
    st.session_state.last_usage_metrics = {}

if "last_context_metrics" not in st.session_state:
    st.session_state.last_context_metrics = {}

if "last_delegation_trace" not in st.session_state:
    st.session_state.last_delegation_trace = ""

if "last_events" not in st.session_state:
    st.session_state.last_events = []


# =========================================================
# Load Config
# =========================================================

try:
    agents_config, tasks_config = load_configs()
except Exception as config_error:
    st.error(f"Config loading failed: {config_error}")
    st.stop()


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.title("⚙️ CrewAI Delegation Settings")

    ollama_base_url = st.text_input(
        "Ollama Base URL",
        value=DEFAULT_OLLAMA_BASE_URL,
    )

    available_models = get_ollama_models(ollama_base_url)

    if available_models:
        default_index = (
            available_models.index(DEFAULT_MODEL_NAME)
            if DEFAULT_MODEL_NAME in available_models
            else 0
        )

        selected_model = st.selectbox(
            "Ollama Model",
            options=available_models,
            index=default_index,
        )

        st.success("Ollama connected")

    else:
        selected_model = st.text_input(
            "Ollama Model Name",
            value=DEFAULT_MODEL_NAME,
            help="Enter the exact model name from `ollama list`.",
        )

        st.warning("Ollama not detected or no local models found")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
    )

    max_context_messages = st.slider(
        "Chat History Messages Sent",
        min_value=2,
        max_value=30,
        value=10,
        step=2,
    )

    context_window_tokens = st.number_input(
        "Model Context Window Tokens",
        min_value=1024,
        max_value=262144,
        value=8192,
        step=1024,
    )

    st.divider()

    st.subheader("🤖 Agents From YAML")

    for key, cfg in agents_config.items():
        with st.expander(cfg.get("name", key)):
            st.write(f"**Key:** `{key}`")
            st.write(f"**Role:** {cfg.get('role')}")
            st.write(f"**Delegation:** `{cfg.get('allow_delegation')}`")
            st.caption(cfg.get("description", ""))

    st.divider()

    st.subheader("📊 Context Metrics")

    metrics = st.session_state.last_context_metrics

    if metrics:
        st.metric(
            "Estimated Input Tokens",
            metrics.get("estimated_total_input_tokens", 0),
        )

        st.metric(
            "Context Usage",
            f"{metrics.get('estimated_context_usage_percent', 0)}%",
        )

        st.progress(
            min(
                metrics.get("estimated_context_usage_percent", 0) / 100,
                1.0,
            )
        )

        with st.expander("Full Context Metrics"):
            st.json(metrics)
    else:
        st.caption("Metrics appear after first message.")

    st.divider()

    st.subheader("🧾 CrewAI Usage Metrics")

    if st.session_state.last_usage_metrics:
        st.json(st.session_state.last_usage_metrics)
    else:
        st.caption("Usage metrics appear after first run.")

    st.divider()

    st.subheader("🧭 Last Activity Events")

    if st.session_state.last_events:
        for event in st.session_state.last_events[-10:]:
            st.write(
                f"{event_icon(event.get('type'))} "
                f"**{event.get('title')}**"
            )
            st.caption(
                f"{event.get('time')} · {event.get('agent')} · {event.get('detail')[:120]}"
            )
    else:
        st.caption("Events appear after first run.")

    st.divider()

    st.subheader("📜 Delegation Trace")

    if st.session_state.last_delegation_trace:
        with st.expander("View CrewAI Verbose Trace"):
            st.text_area(
                label="CrewAI Delegation Logs",
                value=st.session_state.last_delegation_trace,
                height=300,
            )
    else:
        st.caption("Delegation trace appears after first run.")

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_usage_metrics = {}
        st.session_state.last_context_metrics = {}
        st.session_state.last_delegation_trace = ""
        st.session_state.last_events = []
        st.rerun()


# =========================================================
# Main UI
# =========================================================

st.title("🤖 CrewAI Native Delegation Chat")

st.markdown(
    """
### Team Setup

This app uses **CrewAI hierarchical delegation**.

- **Supervisor Agent** is the manager.
- **Data Scientist Agent** handles ML, AI, statistics, forecasting, GenAI, and modeling.
- **Data Analyst Agent** handles SQL, dashboards, KPIs, EDA, Excel, and reporting.
- The activity panel shows high-level events such as thinking, reasoning, delegating, and completion.
"""
)

st.info(
    f"Running with Ollama model `{selected_model}` at `{ollama_base_url}`"
)


# =========================================================
# Render Existing Messages
# =========================================================

for message in st.session_state.messages:
    role = message.get("role")
    content = message.get("content")
    agent_name = message.get("agent_name", "")

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)

    elif role == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            if agent_name:
                st.markdown(f"**{agent_name}**")
            st.markdown(content)


# =========================================================
# Chat Input
# =========================================================

user_prompt = st.chat_input("Ask your analytics crew...")

if user_prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    chat_history = build_chat_history(
        messages=st.session_state.messages,
        max_messages=max_context_messages,
    )

    context_metrics = calculate_context_metrics(
        chat_history=chat_history,
        user_prompt=user_prompt,
        agents_config=agents_config,
        tasks_config=tasks_config,
        context_window_tokens=context_window_tokens,
    )

    st.session_state.last_context_metrics = context_metrics

    run_events: List[Dict[str, Any]] = []

    add_event(
        run_events,
        "thinking",
        "User message received",
        "Preparing chat history and context metrics.",
        "Streamlit UI",
    )

    add_event(
        run_events,
        "reasoning",
        "Context window estimated",
        (
            f"Estimated input tokens: {context_metrics['estimated_total_input_tokens']} "
            f"({context_metrics['estimated_context_usage_percent']}% of configured window)."
        ),
        "Streamlit UI",
    )

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("**Supervisor Agent**")

        # Use st.empty() so the timeline is replaced on every refresh,
        # instead of duplicated repeatedly.
        timeline_placeholder = st.empty()
        render_event_timeline(run_events, timeline_placeholder)

        assistant_response: Optional[str] = None
        assistant_error: Optional[str] = None

        with st.status(
            "Running CrewAI hierarchical delegation...",
            expanded=True,
        ) as status:
            st.write("💭 Thinking: Supervisor is preparing the task.")
            st.write("🧠 Reasoning: Supervisor will inspect intent and chat history.")
            st.write("🔁 Delegating: CrewAI can assign work to specialist agents.")
            st.write("⚙️ Executing: Ollama model will generate the final answer.")
            st.write(
                f"Estimated input tokens: "
                f"{context_metrics['estimated_total_input_tokens']}"
            )
            st.write(
                f"Estimated context usage: "
                f"{context_metrics['estimated_context_usage_percent']}%"
            )

            try:
                start_time = time.time()

                response, usage_metrics, delegation_trace = run_delegation_crew(
                    user_prompt=user_prompt,
                    chat_history=chat_history,
                    agents_config=agents_config,
                    tasks_config=tasks_config,
                    model_name=selected_model,
                    base_url=ollama_base_url,
                    temperature=temperature,
                    events=run_events,
                    timeline_placeholder=timeline_placeholder,
                )

                elapsed_seconds = round(time.time() - start_time, 2)

                add_event(
                    run_events,
                    "completed",
                    "Response ready",
                    f"Total execution time: {elapsed_seconds} seconds.",
                    "Streamlit UI",
                )

                render_event_timeline(run_events, timeline_placeholder)

                st.session_state.last_usage_metrics = usage_metrics
                st.session_state.last_delegation_trace = delegation_trace
                st.session_state.last_events = run_events

                assistant_response = response

                status.update(
                    label="CrewAI delegation completed",
                    state="complete",
                    expanded=False,
                )

            except Exception as e:
                add_event(
                    run_events,
                    "failed",
                    "CrewAI delegation failed",
                    str(e),
                    "System",
                )

                render_event_timeline(run_events, timeline_placeholder)

                st.session_state.last_events = run_events

                status.update(
                    label="CrewAI delegation failed",
                    state="error",
                    expanded=True,
                )

                assistant_error = f"""
CrewAI hierarchical delegation failed.

```text
{str(e)}
```

Check these:

```bash
ollama list
ollama serve
ollama pull {selected_model}
```

Also confirm these files exist:

```bash
config/agents.yaml
config/tasks.yaml
```

If Ollama is slow, reduce `max_iter` in `config/agents.yaml`.

Example:

```yaml
supervisor_agent:
  max_iter: 4
```
"""

        # IMPORTANT:
        # Final answer is rendered OUTSIDE st.status().
        # If it is rendered inside st.status() and the status is collapsed,
        # the user will only see traces/loaders and miss the final response.
        if assistant_response:
            st.markdown("---")
            st.markdown("### Final Response")
            st.markdown(assistant_response)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "agent_name": "Supervisor Agent",
                    "content": assistant_response,
                }
            )

        elif assistant_error:
            st.error(assistant_error)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "agent_name": "System",
                    "content": assistant_error,
                }
            )