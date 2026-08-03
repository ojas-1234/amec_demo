# AMEC Control Center — Agentic Portfolio Dashboard

A decision-support dashboard for a marine-energy R&D portfolio. You give it a plain-English command; an LLM figures out which projects, sites, or deliverables to update, executes the change through function-calling tools, and writes back a short DOE/RPPR-style summary of what changed and why.

Built as a client demo. The data is **synthetic**, modeled on the structure of an Atlantic Marine Energy Center (AMEC) DOE award — it is illustrative, not real program data.

[FILL: screenshot or demo GIF — typing a command like "mark which projects should be killed, reshaped, or continued" and watching the tables + summary update]

## What it does

The dashboard tracks three linked tables — **projects**, candidate **sites**, and DOE **deliverables** — backed by CSV files. Instead of editing rows by hand, you describe the outcome you want:

> "Evaluate all projects and mark which should be killed, reshaped, or continued."

The agent decides which records the request affects, calls the right tool to update them (status, decision, risk/suitability score, appended notes), saves the change back to CSV, and returns a concise narrative summary in the tone of a DOE deliverable.

## How it works / tech stack

- **Python + Streamlit** for the UI (natural-language command box, plus tabs showing the live Projects / Sites / Deliverables tables).
- **OpenAI Chat Completions with function calling** (`gpt-4.1-mini`) as the orchestration layer.
- Three tools operate on **pandas** DataFrames and persist to CSV:
  - `update_project` — status, decision, `failure_risk_score`, notes
  - `update_site` — `suitability_score`, recommendation, notes
  - `update_deliverable` — status, `risk_score`, notes
- The current contents of all three tables are injected into the model's context each turn, so it only references IDs that actually exist. IDs are matched case-insensitively, and notes are appended rather than overwritten.
- A two-step tool loop: the model requests tool calls → tools run and return results → the model produces the final human-readable summary.

## Running it locally

`app_public.py` is the shareable version — it reads the API key from `st.secrets` or the `OPENAI_API_KEY` environment variable.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run app_public.py
```

## My role

Sole author. I designed the agent's tool schema, the CSV-backed data model, the "inject full table state each turn" approach that keeps the model grounded to real IDs, and the Streamlit interface.
