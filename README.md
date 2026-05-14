# 🎯 PM Assistant Copilot

> AI-Powered Six Sigma Assistant with RAG, LangGraph, DMAIC & FMEA

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-green.svg)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3-orange.svg)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📊 Overview

**PM Assistant Copilot** is an enterprise-grade Six Sigma analysis tool that combines **DMAIC methodology**, **FMEA risk assessment**, and **RAG-powered knowledge retrieval** into a single Streamlit application. Powered by LangGraph orchestration and Groq's LLaMA 3.3 70B model, it delivers professional quality analysis with real citations from your knowledge base.

### ✨ Key Features

- 🔄 **3 Analysis Paths**: DMAIC, FMEA, or Hybrid (DMAIC + FMEA)
- 🧠 **AI-Powered**: LLM-driven analysis for all Six Sigma phases
- 📚 **RAG Integration**: Cites sources from your PMBOK/PDF knowledge base
- 📊 **Rich Visualizations**: Sigma gauge, Pareto charts, SPC charts, Impact/Effort matrix
- ⚠️ **FMEA Analysis**: Full RPN scoring with poka-yoke suggestions
- 📋 **Project Charter**: Auto-generated with CTQs, SIPOC, SWOT
- 🎛️ **Dynamic Tabs**: UI adapts to selected analysis path
- 📥 **Export Options**: JSON, Excel reports with citations
- 🏥 **System Health Dashboard**: Token usage, rate limit monitoring
- 🎨 **Modern UI**: Gradient theme, glass-morphism cards, responsive design

---

## 🏗️ Architecture

┌─────────────────────────────────────────────────────────┐
│ STREAMLIT FRONTEND │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│ │ Define │ │ Measure │ │ Analyze │ │ Improve │ │
│ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ │
│ └───────────┴──────────┴───────────┘ │
│ │ │
│ ┌────┴────┐ │
│ │ Control │ │
│ └────┬────┘ │
│ │ │
│ ┌─────────┴─────────┐ │
│ │ FMEA / Export │ │
│ └───────────────────┘ │
└──────────────────────┬──────────────────────────────────┘
│
┌──────────────────────┴──────────────────────────────────┐
│ LANGGRAPH BACKEND │
│ ┌──────────────────────────────────────────────────┐ │
│ │ Router → RAG Retrieve → Vector Store → Nodes │ │
│ │ │ │
│ │ DMAIC: Define → Measure → Analyze → Improve → │ │
│ │ Control → Excel → Citations → Display │ │
│ │ │ │
│ │ FMEA: FMEA → Poka-Yoke → Excel → Display │ │
│ │ │ │
│ │ Hybrid: DMAIC → Hybrid → FMEA → Export │ │
│ └──────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────┘
│
┌──────────────────────┴──────────────────────────────────┐
│ KNOWLEDGE BASE (RAG) │
│ ┌──────────────────────────────────────────────────┐ │
│ │ PDF Loader → Text Splitter → FAISS Vector Store │ │
│ │ │ │
│ │ Sources: PMBOK Guide, Six Sigma Handbooks │ │
│ └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Groq API Key ([Get one here](https://console.groq.com))
- PDF documents for knowledge base (optional)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd PM_Assistant_Copilot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
`
```
## Configuration
1. Create a .env file:
``` env
GROQ_API_KEY=your_groq_api_key_here
```
2. Add PDFs to your knowledge base (optional):
- Place PDF files in a directory
- Update `PM_BOOKS_PATH` in `backend.py`
### Run the Application
```bash
streamlit run frontend.py
```
The app will open at `http://localhost:8501`
## 📖 Usage Guide
1. Select Analysis Path
- 📈 DMAIC Only: Process improvement for existing problems
- ⚠️ FMEA Only: Risk assessment for new designs/processes
- 🔄 Hybrid: Combined DMAIC + FMEA analysis

2. Enter Problem Details
- Problem statement (or use a template)
- Customer/Stakeholder
- Baseline & Target defect rates
- Process steps (one per line)
- CTQs (Critical to Quality metrics)

3. Configure Settings (Sidebar)
- Model Configuration: Temperature, RAG chunks, max tokens
- Specification Limits: USL, LSL, Standard Deviation
- Knowledge Base: View loaded sources

4. Run Analysis

    Click 🚀 START ANALYSIS and monitor progress through 6 phases.

5. Review Results
Navigate through dynamic tabs:
- Define: Project Charter, CTQs, SIPOC, SWOT
- Measure: Baseline metrics, Sigma gauge
- Analyze: Root causes, Pareto chart, Fishbone diagram
- Improve: Solutions table, Impact/Effort matrix
- Control: Cpk, SPC chart, Control plan
- FMEA: RPN scores, Poka-Yoke suggestions
- Sources: Cited references from knowledge base

6. Export
- 📥 JSON Report
- 📊 Excel Report (auto-generated)

## 📁 Project Structure
```text
PM_Assistant_Copilot/
├── backend.py              # LangGraph backend with DMAIC/FMEA logic
├── frontend_V1_rev.2.py    # Streamlit UI (See newest Verison)
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── knowledge_base/         # PDF documents for RAG
├── DMAIC_V1.db            # SQLite checkpoint database
├── fmea_report.csv/xlsx   # Generated FMEA exports
├── dmaic_report.xlsx      # Generated DMAIC exports
└── README.md              # This file
```

## 🔧 Dependencies
```test
"ipykernel",
"streamlit",
"pandas",
"plotly",
"numpy",
"langchain-core",
"langgraph-checkpoint-sqlite",
"langgraph",
"dotenv",
"langchain-groq",
"langchain-huggingface",
"langchain_community",
"sentence-transformers",
"pypdf",
"faiss-cpu",
"openpyxl"
```
## 🎯 Features in Detail
|DMAIC| Methodology|
|-----|-------------|
|Phase	|Output
|Define	|Project Charter, CTQ Tree, SIPOC, SWOT|
|Measure	|Baseline metrics, Sigma level, DPU, Data collection plan|
|Analyze	|5 Whys, Fishbone (6Ms), Pareto chart|
|Improve	|Prioritized solutions, Impact/Effort matrix|
|Control	|Cpk, SPC limits, Control plan, Sustained status|

## FMEA Analysis
- Process step evaluation
- Failure mode identification
- Severity, Occurrence, Detection scoring (1-10)
- RPN calculation (S × O × D)
- Poka-Yoke mistake-proofing suggestions
- Critical risk flagging (RPN ≥ 200)

## RAG Knowledge Base
- PDF document loading
- Recursive text splitting
- FAISS vector embeddings
- Semantic search retrieval
- Source citation tracking

## 🛡️ Error Handling
- Rate Limit Detection: Automatic detection of Groq API limits with wait time estimation
- Graceful Fallbacks: Demo mode when backend is unavailable
- Phase Fallbacks: Each DMAIC phase has default values if LLM fails
- Validation: Input validation before analysis execution
- Session Persistence: SQLite checkpointing for state recovery

## 🚢 Deployment
### Streamlit Cloud
```bash
# Add to secrets.toml
GROQ_API_KEY = "your_key"
```
### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "frontend.py"]
```
## 📝 License
MIT License - see LICENSE file

## 🤝 Contributing
Contributions welcome! Please open an issue or pull request.

## 📧 Support
- Rate Limits: Free tier has 100k tokens/day. Upgrade at Groq Console
- Issues: Open a GitHub issue
- Documentation: Check inline code comments

