#  AMEC Control Center — Agentic Dashboard

The **AMEC Control Center** is an AI-powered decision-support dashboard for the Atlantic Marine Energy Center.  
It uses **OpenAI tool calling**, **Streamlit**, and structured CSV datasets to evaluate:

- Site failure risk  
- Site suitability  
- Deliverable delays  
- Project status  
- Portfolio-level operational decisions  

Everything is driven through **natural language**, and the AMEC Agent automatically chooses and executes the correct analytical tool based on the user's query.

---

## 🚀 Features

### ✅ Natural Language Interface  
Ask questions like:
> “Which site is riskiest?”

The AMEC Brain automatically invokes the necessary tool.

### ✅ Agentic Tool Execution  
Three Python tools operate on the CSV data:

1. **evaluate_failure_risk(site_id)**  
2. **evaluate_site_suitability(site_id)**  
3. **check_deliverable_risk(project_id)**  

These tools read from:
- `projects.csv`  
- `sites.csv`  
- `deliverables.csv`

### ✅ DOE/RPPR-style Summaries  
The AI generates structured, DOE-appropriate narrative summaries when relevant.

### ✅ Dashboard UI (Streamlit)  
A clean UI for:
- Query input  
- Tool output  
- Debug messages (optional)  

---

## 📦 Installation & Setup

### 1. Project Structure

```
amec_dashboard/
│ app.py
│ projects.csv
│ sites.csv
│ deliverables.csv
```

---

### 2. Install Dependencies

Run in terminal:

```bash
pip install streamlit openai pandas python-dotenv
```

---

### 3. Set Your OpenAI API Key

#### Temporary (recommended for demo):
```bash
export OPENAI_API_KEY="sk-..."
```

#### Permanent:
```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc
source ~/.zshrc
```

---

## ▶️ Running the Dashboard

From inside the project folder:

```bash
cd amec_dashboard
streamlit run app.py
```

Then open:

👉 http://localhost:8501

---

## 🧠 How the Agent Works

1. User enters a natural-language question.  
2. The OpenAI model interprets intent.  
3. It selects the correct tool using `tool_choice="auto"`.  
4. It extracts the correct IDs (S001, P002, etc.).  
5. The Python tool executes using the CSV data.  
6. The output is returned as structured JSON or a narrative summary.

---

## 💬 Best Demo Prompts

These prompts show the tool-calling and analysis clearly:

### 🔥 Risk & Sites
- “Which site has the highest failure risk and why?”
- “Evaluate failure on S001.”
- “Rank all sites from highest to lowest risk.”

### 🔥 Suitability & Deployment
- “Which site is most suitable for deployment?”
- “Compare S001 and S003 for suitability.”
- “What is the best tradeoff between risk and suitability?”

### 🔥 Deliverables & Project Health
- “Which deliverables are behind for P002?”
- “Which project is most likely to slip schedule?”
- “Summarize deliverable risks across the portfolio.”

### 🔥 Director-Level Insights
- “Give me a deployment recommendation like I’m the AMEC director.”
- “What should AMEC prioritize this quarter?”

---

## 🛠 Troubleshooting

### ❌ API key error  
Run:
```bash
export OPENAI_API_KEY="sk-..."
```

### ❌ Browser shows a directory instead of UI  
You're opening a file path — open:
👉 http://localhost:8501

### ❌ Tools not triggering  
Enable debugging inside `app.py`:
```python
st.write("DEBUG RAW MODEL RESPONSE:", msg)
```

This reveals whether the LLM attempted a tool call.
