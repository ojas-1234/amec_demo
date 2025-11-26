"""
AMEC Control Center - Unified Dashboard
Combines project/site/deliverable tracking with charter document editing.

Run with: streamlit run app.py
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- CONFIG ----------

MODEL = "gpt-4.1-mini"
def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("OPENAI_API_KEY", "")

client = OpenAI(api_key=get_api_key())

PROJECTS_CSV = "projects.csv"
SITES_CSV = "sites.csv"
DELIVERABLES_CSV = "deliverables.csv"
CHARTER_FILE = "charter.md"
VERSIONS_DIR = "charter_versions"


# =============================================================================
# PART 1: CSV DATA OPERATIONS (Original functionality)
# =============================================================================

def load_data():
    if "projects_df" not in st.session_state:
        df = pd.read_csv(PROJECTS_CSV)
        df.columns = df.columns.str.lower()
        st.session_state.projects_df = df
    if "sites_df" not in st.session_state:
        df = pd.read_csv(SITES_CSV)
        df.columns = df.columns.str.lower()
        st.session_state.sites_df = df
    if "deliverables_df" not in st.session_state:
        df = pd.read_csv(DELIVERABLES_CSV)
        df.columns = df.columns.str.lower()
        st.session_state.deliverables_df = df


def save_data():
    st.session_state.projects_df.to_csv(PROJECTS_CSV, index=False)
    st.session_state.sites_df.to_csv(SITES_CSV, index=False)
    st.session_state.deliverables_df.to_csv(DELIVERABLES_CSV, index=False)


def tool_update_project(arguments: dict):
    df = st.session_state.projects_df
    pid = arguments.get("project_id")
    if pid is None:
        return {"ok": False, "error": "project_id is required"}

    mask = df["project_id"].str.upper() == str(pid).upper()
    if not mask.any():
        return {"ok": False, "error": f"Project {pid} not found"}

    row_idx = df.index[mask][0]

    if "status" in arguments and arguments["status"] is not None:
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
    df = st.session_state.sites_df
    sid = arguments.get("site_id")
    if sid is None:
        return {"ok": False, "error": "site_id is required"}

    mask = df["site_id"].str.upper() == str(sid).upper()
    if not mask.any():
        return {"ok": False, "error": f"Site {sid} not found. Available: {df['site_id'].tolist()}"}

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


# =============================================================================
# PART 2: CHARTER DOCUMENT OPERATIONS (New functionality)
# =============================================================================

def ensure_versions_dir():
    Path(VERSIONS_DIR).mkdir(exist_ok=True)


def load_charter() -> str:
    if "charter_content" not in st.session_state:
        if Path(CHARTER_FILE).exists():
            st.session_state.charter_content = Path(CHARTER_FILE).read_text()
        else:
            st.session_state.charter_content = "# Empty Charter\nNo charter loaded."
    return st.session_state.charter_content


def save_charter(content: str, updated_by: str = "System"):
    ensure_versions_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_file = Path(VERSIONS_DIR) / f"charter_{timestamp}.md"
    
    if Path(CHARTER_FILE).exists():
        current_content = Path(CHARTER_FILE).read_text()
        version_file.write_text(current_content)
    
    content = update_charter_metadata(content, updated_by)
    Path(CHARTER_FILE).write_text(content)
    st.session_state.charter_content = content
    
    return timestamp


def update_charter_metadata(content: str, updated_by: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    content = re.sub(
        r'\| Last Updated By \| .* \|',
        f'| Last Updated By | {updated_by} |',
        content
    )
    content = re.sub(
        r'\| Last Updated On \| .* \|',
        f'| Last Updated On | {now} |',
        content
    )
    
    version_match = re.search(r'\| Version \| ([\d.]+) \|', content)
    if version_match:
        old_version = version_match.group(1)
        parts = old_version.split('.')
        parts[-1] = str(int(parts[-1]) + 1)
        new_version = '.'.join(parts)
        content = re.sub(
            r'\| Version \| [\d.]+ \|',
            f'| Version | {new_version} |',
            content
        )
    
    return content


def parse_charter_tasks(content: str) -> dict:
    tasks = {}
    task_pattern = r'^## (Task [\d.]+):?\s*(.*)$'
    lines = content.split('\n')
    
    current_task = None
    current_lines = []
    
    for i, line in enumerate(lines):
        match = re.match(task_pattern, line)
        if match:
            if current_task:
                tasks[current_task['id']] = {
                    'id': current_task['id'],
                    'title': current_task['title'],
                    'content': '\n'.join(current_lines),
                    'start_line': current_task['start_line']
                }
            current_task = {
                'id': match.group(1),
                'title': match.group(2).strip(),
                'start_line': i
            }
            current_lines = [line]
        elif current_task:
            if line.startswith('## ') and not re.match(task_pattern, line):
                tasks[current_task['id']] = {
                    'id': current_task['id'],
                    'title': current_task['title'],
                    'content': '\n'.join(current_lines),
                    'start_line': current_task['start_line']
                }
                current_task = None
                current_lines = []
            else:
                current_lines.append(line)
    
    if current_task:
        tasks[current_task['id']] = {
            'id': current_task['id'],
            'title': current_task['title'],
            'content': '\n'.join(current_lines),
            'start_line': current_task['start_line']
        }
    
    return tasks


def get_charter_task_list(content: str) -> list:
    tasks = parse_charter_tasks(content)
    return [{'id': t['id'], 'title': t['title']} for t in tasks.values()]


def read_charter_task(content: str, task_id: str) -> dict:
    tasks = parse_charter_tasks(content)
    
    normalized = task_id.strip()
    if not normalized.lower().startswith('task'):
        normalized = f"Task {normalized}"
    
    for tid, task in tasks.items():
        if tid.lower() == normalized.lower():
            return {'found': True, 'task': task}
    
    for tid, task in tasks.items():
        if normalized.lower() in tid.lower() or tid.lower() in normalized.lower():
            return {'found': True, 'task': task}
    
    return {'found': False, 'error': f"Task '{task_id}' not found. Available: {list(tasks.keys())}"}


def update_charter_task_section(content: str, task_id: str, section: str, new_value: str) -> dict:
    result = read_charter_task(content, task_id)
    if not result['found']:
        return {'ok': False, 'error': result['error']}
    
    task = result['task']
    task_content = task['content']
    updated_task_content = task_content
    
    section_lower = section.lower().strip()
    
    status_fields = {
        'progress': r'\| Progress \| .* \|',
        'schedule_status': r'\| Schedule Status \| .* \|',
        'budget_status': r'\| Budget Status \| .* \|',
        'risk_level': r'\| Risk Level \| .* \|'
    }
    
    if section_lower in status_fields:
        pattern = status_fields[section_lower]
        field_name = section_lower.replace('_', ' ').title()
        replacement = f'| {field_name} | {new_value} |'
        updated_task_content = re.sub(pattern, replacement, task_content, flags=re.IGNORECASE)
    
    elif section_lower in ['activities', 'project activities completed']:
        updated_task_content = replace_charter_section_content(task_content, '### Project Activities Completed', new_value)
    
    elif section_lower in ['challenges', 'current challenges']:
        updated_task_content = replace_charter_section_content(task_content, '### Current Challenges', new_value)
    
    elif section_lower in ['plans', 'plans for next quarter']:
        updated_task_content = replace_charter_section_content(task_content, '### Plans for Next Quarter', new_value)
    
    elif section_lower == 'notes':
        updated_task_content = replace_charter_section_content(task_content, '### Notes', new_value)
    
    elif section_lower == 'overview':
        updated_task_content = replace_charter_section_content(task_content, '### Overview', new_value)
    
    else:
        return {'ok': False, 'error': f"Unknown section: {section}"}
    
    new_content = content.replace(task_content, updated_task_content)
    return {'ok': True, 'updated_content': new_content, 'task_id': task['id']}


def replace_charter_section_content(task_content: str, section_header: str, new_value: str) -> str:
    lines = task_content.split('\n')
    result_lines = []
    in_target_section = False
    
    for line in lines:
        if line.strip() == section_header:
            result_lines.append(line)
            result_lines.append(new_value)
            in_target_section = True
        elif in_target_section and (line.startswith('### ') or line.startswith('## ') or line.strip() == '---'):
            in_target_section = False
            result_lines.append('')
            result_lines.append(line)
        elif not in_target_section:
            result_lines.append(line)
    
    return '\n'.join(result_lines)


def append_to_charter_task_section(content: str, task_id: str, section: str, text_to_append: str) -> dict:
    result = read_charter_task(content, task_id)
    if not result['found']:
        return {'ok': False, 'error': result['error']}
    
    task = result['task']
    task_content = task['content']
    
    section_lower = section.lower().strip()
    section_map = {
        'activities': '### Project Activities Completed',
        'project activities completed': '### Project Activities Completed',
        'challenges': '### Current Challenges',
        'current challenges': '### Current Challenges',
        'plans': '### Plans for Next Quarter',
        'plans for next quarter': '### Plans for Next Quarter',
        'notes': '### Notes'
    }
    
    if section_lower not in section_map:
        return {'ok': False, 'error': f"Cannot append to section: {section}"}
    
    header = section_map[section_lower]
    
    lines = task_content.split('\n')
    result_lines = []
    in_target_section = False
    appended = False
    
    for line in lines:
        result_lines.append(line)
        if line.strip() == header:
            in_target_section = True
        elif in_target_section and (line.startswith('### ') or line.startswith('## ') or line.strip() == '---'):
            if not appended:
                result_lines.insert(-1, text_to_append)
                appended = True
            in_target_section = False
    
    if in_target_section and not appended:
        result_lines.append(text_to_append)
    
    updated_task_content = '\n'.join(result_lines)
    new_content = content.replace(task_content, updated_task_content)
    
    return {'ok': True, 'updated_content': new_content, 'task_id': task['id']}


def list_charter_versions() -> list:
    ensure_versions_dir()
    versions = []
    for f in sorted(Path(VERSIONS_DIR).glob("charter_*.md"), reverse=True):
        timestamp = f.stem.replace("charter_", "")
        try:
            dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            versions.append({
                'file': f.name,
                'timestamp': timestamp,
                'datetime': dt.strftime("%Y-%m-%d %H:%M:%S")
            })
        except ValueError:
            pass
    return versions[:10]


def restore_charter_version(timestamp: str) -> dict:
    version_file = Path(VERSIONS_DIR) / f"charter_{timestamp}.md"
    if not version_file.exists():
        return {'ok': False, 'error': f"Version {timestamp} not found"}
    
    content = version_file.read_text()
    save_charter(content, updated_by="System (restored)")
    return {'ok': True, 'restored': timestamp}


# Charter tool implementations
def tool_list_charter_tasks(arguments: dict) -> dict:
    content = load_charter()
    tasks = get_charter_task_list(content)
    return {'ok': True, 'tasks': tasks}


def tool_read_charter_task(arguments: dict) -> dict:
    task_id = arguments.get('task_id')
    if not task_id:
        return {'ok': False, 'error': 'task_id is required'}
    
    content = load_charter()
    result = read_charter_task(content, task_id)
    
    if result['found']:
        return {'ok': True, 'task': result['task']}
    return {'ok': False, 'error': result['error']}


def tool_update_charter_task(arguments: dict) -> dict:
    task_id = arguments.get('task_id')
    section = arguments.get('section')
    new_value = arguments.get('new_value')
    updated_by = arguments.get('updated_by', 'AI Agent')
    
    if not all([task_id, section, new_value]):
        return {'ok': False, 'error': 'task_id, section, and new_value are required'}
    
    content = load_charter()
    result = update_charter_task_section(content, task_id, section, new_value)
    
    if result['ok']:
        save_charter(result['updated_content'], updated_by)
        return {'ok': True, 'message': f"Updated {section} for {result['task_id']}"}
    return result


def tool_append_to_charter_task(arguments: dict) -> dict:
    task_id = arguments.get('task_id')
    section = arguments.get('section')
    text = arguments.get('text')
    updated_by = arguments.get('updated_by', 'AI Agent')
    
    if not all([task_id, section, text]):
        return {'ok': False, 'error': 'task_id, section, and text are required'}
    
    content = load_charter()
    result = append_to_charter_task_section(content, task_id, section, text)
    
    if result['ok']:
        save_charter(result['updated_content'], updated_by)
        return {'ok': True, 'message': f"Appended to {section} for {result['task_id']}"}
    return result


def tool_list_charter_versions(arguments: dict) -> dict:
    versions = list_charter_versions()
    return {'ok': True, 'versions': versions}


def tool_restore_charter_version(arguments: dict) -> dict:
    timestamp = arguments.get('timestamp')
    if not timestamp:
        return {'ok': False, 'error': 'timestamp is required'}
    return restore_charter_version(timestamp)


# =============================================================================
# PART 3: UNIFIED TOOLS DEFINITION
# =============================================================================

TOOLS = [
    # --- CSV Data Tools (Original) ---
    {
        "type": "function",
        "function": {
            "name": "update_project",
            "description": "Update a project's status, decision, risk score or notes in the CSV data.",
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
            "description": "Update a site's suitability score, recommendation, or notes in the CSV data.",
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
            "description": "Update a deliverable's status, risk score, or notes in the CSV data.",
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
    # --- Charter Document Tools (New) ---
    {
        "type": "function",
        "function": {
            "name": "list_charter_tasks",
            "description": "List all tasks in the project charter document with their IDs and titles.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_charter_task",
            "description": "Read the full content of a specific task from the charter document by ID (e.g., 'Task 9.0').",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "The task ID, e.g., 'Task 9.0' or '9.0'"}
                },
                "required": ["task_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_charter_task",
            "description": "Update a section of a charter task. Sections: progress, schedule_status, budget_status, risk_level, activities, challenges, plans, notes, overview.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "The task ID to update"},
                    "section": {"type": "string", "description": "Section: progress, schedule_status, budget_status, risk_level, activities, challenges, plans, notes, overview"},
                    "new_value": {"type": "string", "description": "The new value or content"},
                    "updated_by": {"type": "string", "description": "Name of person making the update"}
                },
                "required": ["task_id", "section", "new_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "append_to_charter_task",
            "description": "Append content to a charter task section (add bullet points without replacing). Sections: activities, challenges, plans, notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "The task ID"},
                    "section": {"type": "string", "description": "Section: activities, challenges, plans, notes"},
                    "text": {"type": "string", "description": "Text to append (e.g., '- New item')"},
                    "updated_by": {"type": "string", "description": "Name of person making the update"}
                },
                "required": ["task_id", "section", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_charter_versions",
            "description": "List available backup versions of the charter document.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "restore_charter_version",
            "description": "Restore the charter to a previous version by timestamp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Version timestamp (YYYYMMDD_HHMMSS)"}
                },
                "required": ["timestamp"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    # CSV tools
    "update_project": tool_update_project,
    "update_site": tool_update_site,
    "update_deliverable": tool_update_deliverable,
    # Charter tools
    "list_charter_tasks": tool_list_charter_tasks,
    "read_charter_task": tool_read_charter_task,
    "update_charter_task": tool_update_charter_task,
    "append_to_charter_task": tool_append_to_charter_task,
    "list_charter_versions": tool_list_charter_versions,
    "restore_charter_version": tool_restore_charter_version,
}


# =============================================================================
# PART 4: LLM ORCHESTRATION
# =============================================================================

SYSTEM_PROMPT = """
You are the AMEC Control Brain — a unified assistant for the Atlantic Marine Energy Center.

You manage TWO types of data:

1. CSV PORTFOLIO DATA (projects, sites, deliverables tables):
   - update_project(project_id, status, decision, failure_risk_score, notes)
   - update_site(site_id, suitability_score, recommendation, notes)
   - update_deliverable(deliverable_id, status, risk_score, notes)
   Use these for portfolio-level decisions like "kill P001" or "mark S002 as high suitability".

2. CHARTER DOCUMENT (detailed task narratives in markdown):
   - list_charter_tasks() — see all charter tasks
   - read_charter_task(task_id) — read a task's full content
   - update_charter_task(task_id, section, new_value) — replace a section
   - append_to_charter_task(task_id, section, text) — add to a section
   - list_charter_versions() / restore_charter_version(timestamp) — version control
   
   Charter sections: progress, schedule_status, budget_status, risk_level, activities, challenges, plans, notes, overview
   Use these for detailed quarterly updates like "update Task 9.0 progress to 85%" or "add a challenge to Task 4.2".

GUIDELINES:
- For portfolio/CSV operations: use update_project, update_site, update_deliverable
- For charter document edits: use the charter tools
- If user says "task" they likely mean charter; if they say "project P001" they mean CSV
- Always confirm what was changed after updates
- Be concise and DOE-appropriate in summaries
"""


def call_llm_with_tools(user_input: str, debug=False):
    """Unified LLM call with all tools."""
    
    # Load all context
    projects_data = st.session_state.projects_df.to_dict(orient="records")
    sites_data = st.session_state.sites_df.to_dict(orient="records")
    deliv_data = st.session_state.deliverables_df.to_dict(orient="records")
    charter_tasks = get_charter_task_list(load_charter())
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"CSV PROJECTS:\n{json.dumps(projects_data, indent=2)}"},
        {"role": "system", "content": f"CSV SITES:\n{json.dumps(sites_data, indent=2)}"},
        {"role": "system", "content": f"CSV DELIVERABLES:\n{json.dumps(deliv_data, indent=2)}"},
        {"role": "system", "content": f"CHARTER TASKS:\n{json.dumps(charter_tasks, indent=2)}"},
        {"role": "user", "content": user_input},
    ]
    
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    
    msg = resp.choices[0].message
    
    if debug:
        st.write("DEBUG - Raw response:", resp)
    
    if not msg.tool_calls:
        return msg.content
    
    # Execute tool calls
    tool_results = []
    for tc in msg.tool_calls:
        fn_name = tc.function.name
        args = json.loads(tc.function.arguments or "{}")
        
        if fn_name in TOOL_FUNCTIONS:
            result = TOOL_FUNCTIONS[fn_name](args)
        else:
            result = {"ok": False, "error": f"Unknown tool {fn_name}"}
        
        tool_results.append({
            "tool_call_id": tc.id,
            "name": fn_name,
            "content": json.dumps(result),
        })
    
    # Follow-up with tool results
    follow_messages = messages + [msg]
    for tr in tool_results:
        follow_messages.append({
            "role": "tool",
            "tool_call_id": tr["tool_call_id"],
            "name": tr["name"],
            "content": tr["content"],
        })
    
    resp2 = client.chat.completions.create(
        model=MODEL,
        messages=follow_messages,
    )
    
    return resp2.choices[0].message.content


# =============================================================================
# PART 5: STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(page_title="AMEC Control Center", layout="wide")
    st.title("🌊 AMEC Control Center")
    st.caption("Unified dashboard: Portfolio tracking + Charter document editing")
    
    # Load data
    load_data()
    load_charter()
    
    # --- NLP Interface ---
    st.subheader("Ask the AMEC Brain")
    
    user_query = st.text_area(
        "Natural language command",
        placeholder="Examples:\n• Which project has the highest risk?\n• Update Task 9.0 progress to 85% complete\n• Add a challenge to Task 4.2: NEPA delays extended\n• Mark P001 decision as Reshape",
        height=100,
    )
    
    col_run, col_debug = st.columns([1, 4])
    with col_run:
        run_button = st.button("Run", type="primary")
    with col_debug:
        debug_mode = st.checkbox("Debug mode")
    
    response_text = None
    if run_button and user_query.strip():
        with st.spinner("Thinking..."):
            try:
                response_text = call_llm_with_tools(user_query.strip(), debug=debug_mode)
                # Reload data in case it changed
                st.session_state.pop('charter_content', None)
                load_charter()
            except Exception as e:
                response_text = f"Error: {e}"
    
    if response_text:
        st.markdown("### 🧠 AMEC Brain Response")
        st.write(response_text)
    
    st.divider()
    
    # --- Data Tabs ---
    st.subheader("Portfolio Data & Charter")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Projects", "📍 Sites", "📋 Deliverables", "📝 Charter"])
    
    with tab1:
        st.markdown("#### Projects")
        st.dataframe(st.session_state.projects_df, use_container_width=True)
    
    with tab2:
        st.markdown("#### Sites")
        st.dataframe(st.session_state.sites_df, use_container_width=True)
    
    with tab3:
        st.markdown("#### Deliverables")
        st.dataframe(st.session_state.deliverables_df, use_container_width=True)
    
    with tab4:
        st.markdown("#### Project Charter")
        
        # Charter metadata sidebar
        content = st.session_state.charter_content
        col_meta, col_versions = st.columns([2, 1])
        
        with col_meta:
            version_match = re.search(r'\| Version \| ([\d.]+) \|', content)
            updated_by_match = re.search(r'\| Last Updated By \| (.*) \|', content)
            updated_on_match = re.search(r'\| Last Updated On \| (.*) \|', content)
            
            meta_cols = st.columns(3)
            with meta_cols[0]:
                st.metric("Version", version_match.group(1) if version_match else "N/A")
            with meta_cols[1]:
                st.caption(f"Updated by: {updated_by_match.group(1) if updated_by_match else 'N/A'}")
            with meta_cols[2]:
                st.caption(f"Updated on: {updated_on_match.group(1) if updated_on_match else 'N/A'}")
        
        with col_versions:
            with st.expander("Version History"):
                versions = list_charter_versions()
                if versions:
                    for v in versions[:5]:
                        st.text(f"📄 {v['datetime']}")
                else:
                    st.text("No backups yet")
        
        # Charter task tabs
        tasks = parse_charter_tasks(content)
        if tasks:
            task_tabs = st.tabs([t['id'] for t in tasks.values()])
            for task_tab, (task_id, task) in zip(task_tabs, tasks.items()):
                with task_tab:
                    st.markdown(task['content'])
        else:
            st.markdown(content)
    
    st.divider()
    st.info("💡 The AMEC Brain can update CSV tables AND charter document via natural language.")


if __name__ == "__main__":
    main()
