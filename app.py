import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- CONFIG ----------

MODEL = "gpt-4.1-mini"  # or gpt-4.1 if you want more power
client = OpenAI(api_key="")

PROJECTS_CSV = "projects.csv"
SITES_CSV = "sites.csv"
DELIVERABLES_CSV = "deliverables.csv"


# ---------- DATA LOADING / SAVING ----------

def load_data():
    if "projects_df" not in st.session_state:
        df = pd.read_csv(PROJECTS_CSV)
        df.columns = df.columns.str.lower()  # Convert to lowercase
        st.session_state.projects_df = df
    if "sites_df" not in st.session_state:
        df = pd.read_csv(SITES_CSV)
        df.columns = df.columns.str.lower()  # Convert to lowercase
        st.session_state.sites_df = df
    if "deliverables_df" not in st.session_state:
        df = pd.read_csv(DELIVERABLES_CSV)
        df.columns = df.columns.str.lower()  # Convert to lowercase
        st.session_state.deliverables_df = df


def save_data():
    st.session_state.projects_df.to_csv(PROJECTS_CSV, index=False)
    st.session_state.sites_df.to_csv(SITES_CSV, index=False)
    st.session_state.deliverables_df.to_csv(DELIVERABLES_CSV, index=False)


# ---------- TOOL IMPLEMENTATIONS (LOCAL FUNCTIONS) ----------

def tool_update_project(arguments: dict):
    """
    Update a project row in projects_df.
    Expected args:
      - project_id (str, required)
      - status (str, optional)
      - decision (str, optional)
      - failure_risk_score (number, optional)
      - notes (str, optional, append)
    """
    df = st.session_state.projects_df
    pid = arguments.get("project_id")
    if pid is None:
        return {"ok": False, "error": "project_id is required"}

    mask = df["project_id"].str.upper() == str(pid).upper()
    if not mask.any():
        return {"ok": False, "error": f"Project {pid} not found"}

    row_idx = df.index[mask][0]

    if "status" in arguments and arguments["status"] is not None:
        # if your column name is different, adjust here
        status_col = "schedule_status" if "schedule_status" in df.columns else "status"
        df.at[row_idx, status_col] = arguments["status"]

    if "decision" in arguments and arguments["decision"] is not None:
        if "decision" not in df.columns:
            df["decision"] = ""
        df.at[row_idx, "decision"] = arguments["decision"]

    if "failure_risk_score" in arguments and arguments["failure_risk_score"] is not None:
        col = "failure_risk_score"
        if col not in df.columns:
            df[col] = ""
        df.at[row_idx, col] = arguments["failure_risk_score"]

    if "notes" in arguments and arguments["notes"]:
        if "notes" not in df.columns:
            df["notes"] = ""
        existing = str(df.at[row_idx, "notes"] or "")
        df.at[row_idx, "notes"] = (existing + " | " if existing else "") + arguments["notes"]

    save_data()
    return {"ok": True, "updated_row": df.loc[row_idx].to_dict()}


def tool_update_site(arguments: dict):
    """
    Update a site row in sites_df.
    Expected args:
      - site_id (str, required)
      - suitability_score (number, optional)
      - recommendation (str, optional)
      - notes (str, optional)
    """
    df = st.session_state.sites_df
    sid = arguments.get("site_id")
    if sid is None:
        return {"ok": False, "error": "site_id is required"}

    # Case-insensitive matching
    mask = df["site_id"].str.upper() == str(sid).upper()
    
    if not mask.any():
        return {"ok": False, "error": f"Site {sid} not found in data. Available sites: {df['site_id'].tolist()}"}

    row_idx = df.index[mask][0]

    if "suitability_score" in arguments and arguments["suitability_score"] is not None:
        col = "suitability_score"
        if col not in df.columns:
            df[col] = ""
        df.at[row_idx, col] = arguments["suitability_score"]

    if "recommendation" in arguments and arguments["recommendation"] is not None:
        col = "recommendation"
        if col not in df.columns:
            df[col] = ""
        df.at[row_idx, col] = arguments["recommendation"]

    if "notes" in arguments and arguments["notes"]:
        col = "notes"
        if col not in df.columns:
            df[col] = ""
        existing = str(df.at[row_idx, col] or "")
        df.at[row_idx, col] = (existing + " | " if existing else "") + arguments["notes"]

    save_data()
    return {"ok": True, "updated_row": df.loc[row_idx].to_dict()}


def tool_update_deliverable(arguments: dict):
    """
    Update a deliverable row in deliverables_df.
    Expected args:
      - deliverable_id (str, required)
      - status (str, optional)
      - risk_score (number, optional)
      - notes (str, optional)
    """
    df = st.session_state.deliverables_df
    did = arguments.get("deliverable_id")
    if did is None:
        return {"ok": False, "error": "deliverable_id is required"}

    mask = df["deliverable_id"].str.upper() == str(did).upper()
    if not mask.any():
        return {"ok": False, "error": f"Deliverable {did} not found"}

    row_idx = df.index[mask][0]

    if "status" in arguments and arguments["status"] is not None:
        df.at[row_idx, "status"] = arguments["status"]

    if "risk_score" in arguments and arguments["risk_score"] is not None:
        col = "risk_score"
        if col not in df.columns:
            df[col] = ""
        df.at[row_idx, col] = arguments["risk_score"]

    if "notes" in arguments and arguments["notes"]:
        col = "notes"
        if col not in df.columns:
            df[col] = ""
        existing = str(df.at[row_idx, col] or "")
        df.at[row_idx, col] = (existing + " | " if existing else "") + arguments["notes"]

    save_data()
    return {"ok": True, "updated_row": df.loc[row_idx].to_dict()}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_project",
            "description": "Update a project’s status, decision, risk score or notes in the AMEC dashboard and CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "status": {"type": "string"},
                    "decision": {"type": "string"},
                    "failure_risk_score": {"type": "number"},
                    "notes": {"type": "string"}
                },
                "required": ["project_id"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_site",
            "description": "Update a site’s suitability score, recommendation, or notes in the AMEC dashboard and CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "site_id": {"type": "string"},
                    "suitability_score": {"type": "number"},
                    "recommendation": {"type": "string"},
                    "notes": {"type": "string"}
                },
                "required": ["site_id"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_deliverable",
            "description": "Update a deliverable’s status, risk score, or notes in the AMEC dashboard and CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "deliverable_id": {"type": "string"},
                    "status": {"type": "string"},
                    "risk_score": {"type": "number"},
                    "notes": {"type": "string"}
                },
                "required": ["deliverable_id"]
            }
        },
    },
]


# ---------- LLM ORCHESTRATION ----------

SYSTEM_PROMPT = """
You are the AMEC Control Brain.

You are managing three tables:
- projects_df: AMEC projects and their status, risk and decisions.
- sites_df: sites and their suitability, permitting and recommendations.
- deliverables_df: DOE-style deliverables and their health.

You can call these tools:
- update_project(...)
- update_site(...)
- update_deliverable(...)

When the user asks for:
- kill / reshape / continue → update 'decision', 'failure_risk_score', and 'notes' in projects_df.
- site evaluation → update 'suitability_score', 'recommendation', 'notes' in sites_df.
- deliverable health → update 'status', 'risk_score', 'notes' in deliverables_df.

ALWAYS:
1. Think about what needs to change in the tables.
2. Call the appropriate tools with the correct IDs.
3. After tool calls, give a short, DOE-style narrative summary of what changed and why.
4. Never hallucinate IDs: only use IDs that exist in the tables the user refers to.
"""


def call_llm_with_tools(user_input: str, debug=False):
    """
    One-step tools loop with full data context
    """
    
    # Give the model the FULL current data so it knows exact IDs
    projects_data = st.session_state.projects_df.to_dict(orient="records")
    sites_data = st.session_state.sites_df.to_dict(orient="records")
    deliv_data = st.session_state.deliverables_df.to_dict(orient="records")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": f"COMPLETE PROJECTS DATA:\n{json.dumps(projects_data, indent=2)}"
        },
        {
            "role": "system",
            "content": f"COMPLETE SITES DATA:\n{json.dumps(sites_data, indent=2)}"
        },
        {
            "role": "system",
            "content": f"COMPLETE DELIVERABLES DATA:\n{json.dumps(deliv_data, indent=2)}"
        },
        {"role": "user", "content": user_input},
    ]
    # First call
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    msg = resp.choices[0].message

    if debug:
        st.write("LLM raw response:", resp)

    # If model didn't request tools, just return its content
    if not msg.tool_calls:
        return msg.content

    # Execute tool calls
    tool_results = []
    for tc in msg.tool_calls:
        fn_name = tc.function.name
        args = json.loads(tc.function.arguments or "{}")

        if fn_name == "update_project":
            result = tool_update_project(args)
        elif fn_name == "update_site":
            result = tool_update_site(args)
        elif fn_name == "update_deliverable":
            result = tool_update_deliverable(args)
        else:
            result = {"ok": False, "error": f"Unknown tool {fn_name}"}

        tool_results.append(
            {
                "tool_call_id": tc.id,
                "name": fn_name,
                "content": json.dumps(result),
            }
        )

    # Build follow-up messages with tool results
    follow_messages = messages + [msg]
    for tr in tool_results:
        follow_messages.append(
            {
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "name": tr["name"],
                "content": tr["content"],
            }
        )

    resp2 = client.chat.completions.create(
        model=MODEL,
        messages=follow_messages,
    )
    final_msg = resp2.choices[0].message
    return final_msg.content


# ---------- STREAMLIT UI ----------

def main():
    st.set_page_config(page_title="AMEC Control Center", layout="wide")
    st.title("AMEC Control Center")
    st.caption("Synthetic marine energy portfolio dashboard with an agentic GPT brain.")

    load_data()

    # Top: NLP control
    st.subheader("Ask the AMEC Brain")
    user_query = st.text_area(
        "Natural language command",
        placeholder="e.g. Evaluate all projects and mark which should be killed, reshaped, or continued.",
        height=100,
    )
    col_run, col_debug = st.columns([1, 3])
    with col_run:
        run_button = st.button("Run", type="primary")
    with col_debug:
        debug_mode = st.checkbox("Show debug LLM output", value=False)

    response_text = None
    if run_button and user_query.strip():
        with st.spinner("Thinking..."):
            try:
                response_text = call_llm_with_tools(user_query.strip(), debug=debug_mode)
            except Exception as e:
                response_text = f"Error calling model: {e}"

    if response_text:
        st.markdown("### AMEC Brain Response")
        st.write(response_text)

    # Divider
    st.markdown("---")

    # Tables section
    st.subheader("Portfolio Data")

    tab1, tab2, tab3 = st.tabs(["Projects", "Sites", "Deliverables"])

    with tab1:
        st.markdown("#### Projects")
        st.dataframe(st.session_state.projects_df, use_container_width=True)

    with tab2:
        st.markdown("#### Sites")
        st.dataframe(st.session_state.sites_df, use_container_width=True)

    with tab3:
        st.markdown("#### Deliverables")
        st.dataframe(st.session_state.deliverables_df, use_container_width=True)

    st.info(
        "The GPT brain can update these tables via natural language, "
        "and changes are saved back to the CSV files."
    )


if __name__ == "__main__":
    main()
