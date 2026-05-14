# ============================================================
# 1. Import libs
# ============================================================
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field, field_validator
from typing import TypedDict, Optional, Tuple, Dict, List
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import uuid
from datetime import datetime
import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import json


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# 2. Load env (API keys)
# ============================================================
load_dotenv()
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1000
)
# ============================================================
# 3. Define Schema (Pydantic Models for LLM Output)
# ============================================================
class ProjectCharter(BaseModel):
    problem_statement: str = ""
    goal_statement: str = ""
    scope: str = ""
    business_case: str = ""
    team_members: List[str] = []
    sponsor: str = ""
    success_metrics: List[str] = []
class SIPOC(BaseModel):
    suppliers: List[str] = []
    inputs: List[str] = []
    process_steps: List[str] = []
    outputs: List[str] = []
    customers: List[str] = []
class SWOT(BaseModel):
    strengths: List[str] = []
    weaknesses: List[str] = []
    opportunities: List[str] = []
    threats: List[str] = []
class DefineOutput(BaseModel):
    project_charter: Optional[ProjectCharter] = None
    ctq_list: List[str] = Field(default_factory=list)
    sipoc: Optional[SIPOC] = None
    swot: Optional[SWOT] = None
class MeasureOutput(BaseModel):
    baseline_metric: float = 0.0
    target_metric: float = 0.0
    sigma_level: float = 0.0
    defects_per_unit: float = 0.0
    data_collection_plan: str = ""
    
    @field_validator('baseline_metric', mode='before')
    @classmethod
    def extract_float_from_dict(cls, v):
        if isinstance(v, dict):
            for key, val in v.items():
                if isinstance(val, (int, float)):
                    return float(val)
            return 0.0
        return float(v) if v else 0.0
    
    @field_validator('target_metric', mode='before')
    @classmethod
    def extract_target_float(cls, v):
        if isinstance(v, dict):
            for key, val in v.items():
                if isinstance(val, (int, float)):
                    return float(val)
            return 0.0
        return float(v) if v else 0.0
class AnalyzeOutput(BaseModel):
    root_causes: List[str] = []
    pareto_top_causes: List[Tuple[str, int]] = Field(default_factory=list)
    sources: List[dict] = Field(default_factory=list) 
class Solution(BaseModel):
    description: str = ""
    effort_score: int = 5
    impact_score: int = 5
    estimated_sigma_improvement: float = 0.0
    risks: List[str] = []
    priority_score: float = 0.0

    @field_validator('risks', mode='before')
    @classmethod
    def convert_risks_to_list(cls, v):
        if isinstance(v, str):
            if v.lower() == 'none':
                return []
            return [r.strip() for r in v.split(',')]
        return v if v else []
class ImproveOutput(BaseModel):
    solutions: List[Solution] = []
class ControlOutput(BaseModel):
    cpk_value: float = 0.0
    spc_limits: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    control_plan: str = ""
    is_sustained: bool = False
    
    @field_validator('spc_limits', mode='before')
    @classmethod
    def convert_spc_to_tuple(cls, v):
        if isinstance(v, dict):
            ucl = float(v.get('UCL', v.get('ucl', 0)))
            mean = float(v.get('mean', v.get('Mean', 0)))
            lcl = float(v.get('LCL', v.get('lcl', 0)))
            return (ucl, mean, lcl)
        if isinstance(v, list):
            return tuple(v)
        return v
    
    @field_validator('control_plan', mode='before')
    @classmethod
    def convert_plan_to_string(cls, v):
        if isinstance(v, dict):
            parts = []
            for key, val in v.items():
                if isinstance(val, dict):
                    parts.append(f"{key}: {json.dumps(val)}")
                else:
                    parts.append(f"{key}: {val}")
            return "; ".join(parts)
        return str(v) if v else ""
class FMEARow(BaseModel):
    process_step: str = ""
    failure_mode: str = ""
    effect: str = ""
    severity: int = 5
    cause: str = ""
    occurrence: int = 5
    current_control: str = ""
    detection: int = 5
    rpn: int = 0
    recommended_action: str = ""
    sources: List[dict] = Field(default_factory=list)
class FishboneDiagram(BaseModel):
    head: str
    bones: Dict[str, List[str]] = Field(default_factory=dict)
    
    @field_validator('bones', mode='before')
    @classmethod
    def convert_bones_to_dict(cls, v):
        if isinstance(v, list):
            ms_categories = ["Machine", "Method", "Material", "Measurement", "Mother Nature", "Manpower"]
            result = {}
            for i, item in enumerate(v):
                key = ms_categories[i] if i < len(ms_categories) else f"Category_{i+1}"
                result[key] = [str(item)] if not isinstance(item, list) else item
            return result
        if isinstance(v, dict):
            return v
        return {}     
class FiveWhysResult(BaseModel):
    whys: List[str] = Field(description="List of 6 items: problem + 5 whys + root cause")

# ============================================================
# 4. Define State (TypedDict)
# ============================================================
class DMAICState(TypedDict, total=False):  
    problem_statement: str
    customer: str
    selected_path: str
    baseline_metric: float
    target_metric: float
    ctq_list: list[str]
    process_steps: list[str]
    sigma_level: float
    evidence: list[dict]
    rag_context: list[dict]
    root_causes: list[str]
    solutions: list
    fmea_rows: list
    defect_data: list
    calc_result: float
    defects_per_unit: float
    data_collection_plan: str
    project_charter: Optional[ProjectCharter]
    sipoc: Optional[SIPOC]
    swot: Optional[SWOT]
    # ADD THESE MISSING KEYS:
    cpk_value: float
    spc_limits: Tuple[float, float, float]
    control_plan: str
    is_sustained: bool
    fishbone_json: dict
    pareto_top_causes: list
    selected_solution: Optional[Solution]
    fmea_rows: list
    top_rpn_risks: list
    critical_risks: list
    citations: list
    citation_text: str
    executive_summary: str
    total_sources: int
    excel_report_path: str
    export_data: list
    display_data: dict
    ready_for_display: bool
    formatted_output: str
    phases_completed: list
    usl: float  # Upper Specification Limit
    lsl: float  # Lower Specification Limit
    stdev: float  # Standard Deviation


# ============================================================
# 5. System Prompts
# ============================================================
DEFINE_PROMPT = """
            You are a Lean Six Sigma Define phase expert.
            Problem: {problem}
            Customer: {customer}

            Use this reference knowledge:
            {context}

            Create a Project Charter with: 
                1. Problem statement (SMART)
                2. Goal statement (measurable)
                3. Scope (what's IN and OUT)
                4. Business case (ROI projection)
                5. CTQ list (Critical to Quality - what matters to customer)
                6. SIPOC (Supplier, Input, Process, Output, Customer)
            Return valid JSON.
            """
MEASURE_PROMPT = """You are a Lean Six Sigma Measure phase expert.
Calculate baseline metrics for: {problem}
CTQs: {ctqs}
Reference: {context}

IMPORTANT FORMAT RULES:
- baseline_metric: Must be a SINGLE float number (not an object)
- target_metric: Must be a SINGLE float number (not an object)  
- sigma_level: Must be a SINGLE float number (not an object)
- defects_per_unit: Must be a SINGLE float number (not an object)
- data_collection_plan: Must be a SINGLE string (not an object)

Example of WRONG format (nested objects):
{{"baseline_metric": {{"defect_rate": 0.025}}}} ← DO NOT DO THIS

Example of CORRECT format (flat values):
{{"baseline_metric": 0.025, "target_metric": 0.01}} ← DO THIS

Analyze the problem and CTQs, then return your calculated metrics as flat float values.
Return valid JSON.
"""
ANALYZE_PROMPT = """You are a Lean Six Sigma Analyze phase expert.
For problem: {problem}
Baseline: {baseline}% defects
Reference knowledge:
{context}

Identify 5-7 specific root causes as simple strings.
ALSO provide pareto_top_causes as a list of [cause, percentage] where percentages sum to 100.
Tools available:
    1. Fishbone diagram (6Ms: Machine, Method, Material, Measurement, Mother Nature, Manpower)
    2. 5 Whys (ask "why" 5 times)
    3. Pareto principle (80/20 rule)

Return valid JSON with this EXACT structure:
{{
    "root_causes": ["cause1", "cause2", "cause3", "cause4", "cause5"],
    "pareto_top_causes": [["cause1", 40], ["cause2", 25], ["cause3", 15], ["cause4", 10], ["cause5", 10]]
}} """
IMPROVE_PROMPT = """You are a Lean Six Sigma Improve phase expert.
    Root causes: {root_causes}
    Current: {baseline}% defects, Target: {target}% defects

    Generate 3-5 solutions. Cite best practices from sources.

    Return valid JSON with 'solutions' array containing objects with: description, effort_score, impact_score, estimated_sigma_improvement, risks, priority_score.
"""
CONTROL_PROMPT = """You are a Lean Six Sigma Control phase expert.
CTQs: {ctqs}
Target: {target}%

IMPORTANT FORMAT RULES:
- cpk_value: Must be a SINGLE float number
- spc_limits: MUST be an array of exactly 3 numbers: [UCL, mean, LCL] (NOT an object with keys)
- control_plan: Must be a SINGLE text string (NOT an object with sub-sections)
- is_sustained: Must be a boolean (true or false)

Example of WRONG format:
{{"spc_limits": {{"UCL": 2.5, "mean": 1.5, "LCL": 0.5}}}} ← DO NOT DO THIS
{{"control_plan": {{"defect_rate": {{"target": 2.0}}}}}} ← DO NOT DO THIS

Example of CORRECT format:
{{"spc_limits": [2.5, 1.5, 0.5]}} ← DO THIS (array of 3 numbers)
{{"control_plan": "Monitor process daily..."}} ← DO THIS (single string)

Analyze the CTQs and target, then create your control plan with proper format types.
Return valid JSON.
"""
FMEA_PROMPT = """You are an FMEA (Failure Mode Effects Analysis) expert.
Analyze process step: {step}
Reference FMEA knowledge:
{context}

Return JSON with ALL these fields: process_step, failure_mode, effect, severity(1-10), cause, occurrence(1-10), detection(1-10), current_control.

Example:
{{
    "process_step": "{step}",
    "failure_mode": "specific failure",
    "effect": "consequences",
    "severity": 7,
    "cause": "root cause",
    "occurrence": 5,
    "detection": 4,
    "current_control": "current method"
}}

Return ONLY valid JSON.
"""
FISHBONE_SYSTEM = "Generate potential causes using 6Ms framework."

# ============================================================
# 6. RAG Knowledge Base
# ============================================================
class KnowledgeBase:
    def __init__(self, knowledge_dir: str = "knowledge_base"):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(exist_ok=True)
        self.vectorstore = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
        self.loaded_files = []
        
    def load_from_directory(self, directory_path: str):
        if not os.path.exists(directory_path):
            print(f"❌ Directory not found: {directory_path}")
            return
        
        loader = DirectoryLoader(
            path=directory_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            recursive=True,
            show_progress=True
        )
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} documents")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(documents)
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        else:
            self.vectorstore.add_documents(splits)
        
        self.loaded_files.append(directory_path)
        print(f"✅ Added {len(splits)} chunks to vectorstore")
    
    def retrieve(self, query: str, k: int = 4) -> Tuple[str, List[dict]]:
        if self.vectorstore is None:
            return "", []
        
        docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        sources = [{"source": d.metadata.get("source", "Unknown"), "snippet": d.page_content[:300]} for d in docs]
        return context, sources

kb = KnowledgeBase()

# Load knowledge base
PM_BOOKS_PATH = r"C:\Users\bivor\OneDrive\Desktop\Links\PM books"
if os.path.exists(PM_BOOKS_PATH):
    kb.load_from_directory(PM_BOOKS_PATH)
#-----------------------------------------------------------

# ============================================================
# 7. Helper Functions
# ============================================================
def calculate_sigma(mean: float, usl: float, lsl: float, stdev: float) -> float:
    if stdev == 0:
        return 0.0
    cpu = (usl - mean) / (3 * stdev)
    cpl = (mean - lsl) / (3 * stdev)
    cpk = min(cpu, cpl)
    return round(cpk * 3, 2)

def calculate_cpk(mean: float, usl: float, lsl: float, stdev: float) -> float:
    if any(v is None for v in [mean, usl, lsl, stdev]):
        return None
    if stdev == 0:
        return 0.0
    cpu = (usl - mean) / (3 * stdev)
    cpl = (mean - lsl) / (3 * stdev)
    return round(min(cpu, cpl), 2)

def generate_spc_limits(new_metric: float, baseline_metric: float) -> tuple:
    if new_metric is None or baseline_metric is None:
        return None
    mean = (new_metric + baseline_metric) / 2
    sigma = abs(new_metric - baseline_metric) / 3
    ucl = mean + (3 * sigma)
    lcl = max(0, mean - (3 * sigma))
    return (round(ucl, 2), round(mean, 2), round(lcl, 2))

def _generate_control_plan(ctq_list: list, cpk: float) -> str:
    if not ctq_list:
        ctq_list = ["Process performance"]
    if cpk is None:
        return "Insufficient data for control plan"
    if cpk >= 1.33:
        frequency = "Weekly"
        response = "Monitor trends; investigate if approaching limits"
    elif cpk >= 1.0:
        frequency = "Daily"
        response = "Review daily; take action if near limits"
    else:
        frequency = "Every shift"
        response = "Immediate investigation required"
    return f"Monitor: {', '.join(ctq_list)} | Frequency: {frequency} | Response: {response}"

def _suggest_poka_yoke(row: FMEARow) -> str:
    suggestions = []
    if row.severity >= 9:
        suggestions.append(f"🔴 Add fail-safe for {row.failure_mode}")
    elif row.severity >= 8:
        suggestions.append("🟠 Add fail-safe mechanism")
    if row.occurrence >= 8:
        suggestions.append("🔴 Redesign process")
    elif row.occurrence >= 7:
        suggestions.append("🟠 Automate step")
    if row.detection >= 8:
        suggestions.append("🔴 Add automated inspection")
    elif row.detection >= 7:
        suggestions.append("🟠 Add poka-yoke check")
    if 5 <= row.occurrence < 7:
        suggestions.append("Implement training")
    return " | ".join(suggestions[:3]) if suggestions else "Monitor"



# ============================================================
# 8. Node Functions
# ============================================================

def collect_user_input_node(state: dict) -> dict:
    """Pass through user input from Streamlit"""
    return {
        "problem_statement": state.get("problem_statement", ""),
        "customer": state.get("customer", ""),
        "selected_path": state.get("selected_path", "dmaic"),
        "baseline_metric": state.get("baseline_metric"),
        "target_metric": state.get("target_metric"),
        "ctq_list": state.get("ctq_list", []),
        "process_steps": state.get("process_steps", []),
        "usl": state.get("usl"),
        "lsl": state.get("lsl"),
        "stdev": state.get("stdev")
    }

def langsmith_trace_node(state: dict) -> dict:
    """Initialize LangSmith trace"""
    trace_id = str(uuid.uuid4())
    return {"rag_context": [{"trace_id": trace_id, "timestamp": datetime.now().isoformat()}]}

def router_node(state: dict) -> dict:
    problem = state.get("problem_statement", "").lower()
    
    if any(word in problem for word in ["new", "design", "risk", "what could go wrong"]):
        path = "fmea"
    elif any(word in problem for word in ["and", "also", "introducing"]) and \
         any(word in problem for word in ["defect", "problem", "issue"]):
        path = "hybrid"
    else:
        path = "dmaic"
    
    return {"selected_path": path}

def rag_retrieve_dmaic_node(state: dict) -> dict:
    context, sources = kb.retrieve(f"DMAIC methodology for: {state['problem_statement']}", k=4)
    return {"rag_context": sources, "evidence": [{"context": context}]}

def rag_retrieve_fmea_node(state: dict) -> dict:
    context, sources = kb.retrieve(f"FMEA analysis for: {state['problem_statement']}", k=4)
    return {"rag_context": sources, "evidence": [{"context": context}]}

def rag_retrieve_hybrid_node(state: dict) -> dict:
    context_dmaic, sources_dmaic = kb.retrieve(f"DMAIC for: {state['problem_statement']}", k=3)
    context_fmea, sources_fmea = kb.retrieve(f"FMEA for: {state['problem_statement']}", k=3)
    return {"rag_context": sources_dmaic + sources_fmea, "evidence": [{"context_dmaic": context_dmaic, "context_fmea": context_fmea}]}

def vector_store_node(state: dict) -> dict:
    """Pass through - vector store already accessed in retrieve nodes"""
    return state

# ------------DEFINE-----------    
def define_node(state: dict) -> dict:
    try:
        context = state.get("evidence", [{}])[0].get("context", "")
        result = model.with_structured_output(DefineOutput, method="json_mode").invoke([
            SystemMessage(content=DEFINE_PROMPT.format(
                problem=state['problem_statement'],
                customer=state.get('customer', 'unknown'),
                context=context
            )),
            HumanMessage(content=f"Create a Project Charter for: {state['problem_statement']}")
        ])
        
        # FIX: Use LLM result primarily, fallback to hardcoded if LLM returns None
        return {
            "project_charter": result.project_charter if result and result.project_charter else ProjectCharter(
                problem_statement=state['problem_statement'],
                goal_statement="Reduce defects by 50% in 3 months",
                scope="Production line A only",
                business_case="Save $500K annually",
                team_members=[],
                sponsor="Executive Sponsor",
                success_metrics=["Defect rate < 1%", "Cycle time < 30min"]
            ),
            "ctq_list": result.ctq_list if result and result.ctq_list else ["Defect rate < 1%", "Cycle time < 30min"],
            "sipoc": result.sipoc if result and result.sipoc else SIPOC(
                suppliers=["Raw material vendors"],
                inputs=["Materials", "Labor"],
                process_steps=["Step1", "Step2", "Step3"],
                outputs=["Finished product"],
                customers=["End user"]
            ),
            "swot": result.swot if result and result.swot else SWOT(
                strengths=["Experienced team"],
                weaknesses=["Old equipment"],
                opportunities=["Automation"],
                threats=["Competition"]
            )
        }
    except Exception as e:
        print(f"⚠️ Define LLM failed: {e}")
        return {"project_charter": " ", "ctq_list": state.get("ctq_list", [])}

# ------------MEASURE-----------    
def measure_node(state: dict) -> dict:
    """Measure phase: Calculate baseline metrics"""
    try:
        context, sources = kb.retrieve(f"Six Sigma measurement baseline calculation: {state.get('problem_statement', '')}", k=3)

        result = model.with_structured_output(MeasureOutput, method="json_mode").invoke([
            SystemMessage(content=MEASURE_PROMPT.format(
                problem=state.get('problem_statement', ''),
                ctqs=', '.join(state.get('ctq_list', [])),
                context=context[:2000]
            )),
            HumanMessage(content=f"Calculate metrics for: {state.get('problem_statement', '')}")
        ])
        
        # FIX: Scale percentages properly
        bl = result.baseline_metric
        tg = result.target_metric
        if bl < 1:
            bl = bl * 100
        if tg < 1:
            tg = tg * 100
            
        return {
            "baseline_metric": bl,
            "target_metric": tg,
            "sigma_level": result.sigma_level,
            "defects_per_unit": result.defects_per_unit,
            "data_collection_plan": result.data_collection_plan,
            "rag_context": sources,
            "calc_result": bl / 100 if bl > 0 else 0
        }
    except Exception as e:
        print(f"⚠️ Measure LLM failed: {e}")
        return {
            "baseline_metric": state.get("baseline_metric"), 
            "target_metric": state.get("target_metric"), 
            "sigma_level": None, 
            "defects_per_unit": None, 
            "data_collection_plan": "Collect 50 samples daily",
            "calc_result": None
        }

# ------------ANALYZE-----------    
def analyze_node(state: dict) -> dict:
    """Analyze phase - identify root causes"""
    try:
        context, sources = kb.retrieve(f"Root cause analysis for: {state.get('problem_statement', '')}", k=5)

        result = model.with_structured_output(AnalyzeOutput, method="json_mode").invoke([
            SystemMessage(content=ANALYZE_PROMPT.format(
                problem=state.get('problem_statement', ''),
                baseline=state.get('baseline_metric', 0),
                context=context[:2000]
            )),
            HumanMessage(content=f"Find root causes for: {state.get('problem_statement', '')}")
        ])
    
        # Generate fishbone diagram using LLM
        fishbone = _generate_fishbone(
            state['problem_statement'], 
            state.get('ctq_list', [])
        )
        
        # Run 5 Whys analysis
        five_whys = _run_five_whys(state['problem_statement'])
        
        # Run Pareto analysis from defect data
        pareto = _run_pareto(state.get('defect_data', []))
        
        return {
            "fishbone_json": fishbone,
            "root_causes": five_whys,
            "pareto_top_causes": pareto[:3]
            }
    except Exception as e:
        print(f"❌ Analyze LLM failed: {e}")
        return {
            "root_causes": [],
            "pareto_top_causes": []
        }

def _generate_fishbone(problem: str, ctqs: list) -> dict:
    """Generate a fishbone diagram using LLM - raw JSON approach"""
    
    ctq_text = f"Critical to Quality metrics: {', '.join(ctqs)}" if ctqs else "No specific CTQs provided"
    
    # Use regular invoke with JSON mode instead of structured output
    response = model.invoke([
        SystemMessage(content="""Generate a fishbone diagram for root cause analysis.
Return ONLY valid JSON with this exact structure:
{
    "head": "problem description",
    "bones": {
        "Machine": ["cause1", "cause2"],
        "Method": ["cause1"],
        "Material": ["cause1"],
        "Measurement": ["cause1"],
        "Mother Nature": ["cause1"],
        "Manpower": ["cause1"]
    }
}
The 'bones' field MUST be a dictionary/object with these 6M categories as keys, NOT an array."""),
        HumanMessage(content=f"Problem to analyze: {problem}\n{ctq_text}")
    ])
    
    try:
        # Parse JSON from response
        import json
        content = response.content if hasattr(response, 'content') else str(response)
        # Extract JSON from possible markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())
    except:
        # Fallback
        return {
            "head": problem,
            "bones": {
                "Machine": ["Unknown"],
                "Method": ["Unknown"],
                "Material": ["Unknown"],
                "Measurement": ["Unknown"],
                "Mother Nature": ["Unknown"],
                "Manpower": ["Unknown"]
            }
        }

def _run_five_whys(problem: str) -> list:
    """Run 5 Whys analysis using LLM"""
    
    FIVE_WHYS_SYSTEM = """
    You are a root cause analyst. Perform a 5 Whys analysis.
    Ask "Why?" 5 times to get to the root cause.
    Return a list of 6 items: [original problem, why1, why2, why3, why4, root_cause]
    """
    five_whys_model = model.with_structured_output(FiveWhysResult)
    
    result = five_whys_model.invoke([
        SystemMessage(content=FIVE_WHYS_SYSTEM),
        HumanMessage(content=f"Problem: {problem}")
    ])
    
    return result.whys

def _run_pareto(defect_data: list) -> list:
    """Run Pareto analysis from defect data"""
    
    if not defect_data:
        return [
            ("Equipment malfunction", 45),
            ("Operator error", 25),
            ("Material defect", 15),
            ("Process variation", 10),
            ("Measurement error", 5)
        ]
    
    sorted_data = sorted(defect_data, key=lambda x: x[1], reverse=True)
    total = sum(count for _, count in sorted_data)
    cumulative = 0
    pareto_items = []
    
    for cause, count in sorted_data:
        cumulative += count
        percentage = (cumulative / total) * 100
        pareto_items.append((cause, count, percentage))
        
        if percentage >= 80:
            break
    
    return pareto_items

# ------------IMPROVE-----------    
def improve_node(state: dict) -> dict:
    """Improve phase: Generate solutions"""
    try:
        context, sources = kb.retrieve(f"Six Sigma improvement solutions for: {', '.join(state.get('root_causes', []))}", k=3)
        result = model.with_structured_output(ImproveOutput, method="json_mode").invoke([
            SystemMessage(content=IMPROVE_PROMPT.format(
                root_causes=state.get('root_causes', []),
                baseline=state.get('baseline_metric', 0),
                target=state.get('target_metric', 0),
                context=context[:2000]
            )),
            HumanMessage(content="Generate improvement solutions")
        ])
        
        for sol in result.solutions:
            if isinstance(sol.risks, str):
                sol.risks = [r.strip() for r in sol.risks.split(',')] if sol.risks.lower() != 'none' else []
            sol.priority_score = (sol.impact_score * 2) - sol.effort_score
        
        selected = max(result.solutions, key=lambda x: x.priority_score) if result.solutions else None
        
        return {
            "solutions": result.solutions,
            "selected_solution": selected,
            "rag_context": sources
        }
    except Exception as e:
        print(f"❌ Improve LLM failed: {e}")
        # FIX: Added fallback return
        return {"solutions": [], "selected_solution": None, "rag_context": []}

# ------------CONTROL-----------    
def control_node(state: dict) -> dict:
    """Control phase: Create control plan"""
    try:
        result = model.with_structured_output(ControlOutput, method="json_mode").invoke([
            SystemMessage(content=CONTROL_PROMPT.format(
                ctqs=state.get('ctq_list', []),
                target=state.get('target_metric', 0)
            )),
            HumanMessage(content="Create control plan")
        ])
        
        # Use LLM results if available, otherwise calculate manually
        if result and isinstance(result.cpk_value, (int, float)) and result.cpk_value > 0:
            return {
                "cpk_value": result.cpk_value,
                "spc_limits": result.spc_limits,
                "control_plan": result.control_plan,
                "is_sustained": result.is_sustained
            }
        else:
            new_metric = state.get('baseline_metric')
            baseline = state.get('baseline_metric')
            usl = state.get('usl')
            lsl = state.get('lsl')
            stdev = state.get('stdev')
            cpk = calculate_cpk(mean=new_metric, usl=usl, lsl=lsl, stdev=stdev)
            spc_limits = generate_spc_limits(new_metric, baseline)
            control_plan = _generate_control_plan(state.get('ctq_list', []), cpk)
            return {
                "cpk_value": cpk,
                "spc_limits": spc_limits,
                "control_plan": control_plan,
                "is_sustained": cpk >= 1.33 if cpk is not None else None
            }
    except Exception as e:
        print(f"⚠️ Control LLM failed: {e}")
        return {
            "cpk_value": None, 
            "spc_limits": None, 
            "control_plan": None, 
            "is_sustained": None
        }

#----------------
def fmea_node(state: dict) -> dict:
    """FMEA analysis node"""
    steps = state.get('process_steps', [])
    if not steps:
        steps = ["Process Step 1", "Process Step 2", "Process Step 3", "Process Step 4", "Process Step 5"]
    
    fmea_rows = []
    for step in steps:
        try:
            context, sources = kb.retrieve(f"FMEA analysis for process step: {step}", k=3)
            
            row = model.with_structured_output(FMEARow, method="json_mode").invoke([
                SystemMessage(content=FMEA_PROMPT.format(step=step, context=context[:2000])),
                HumanMessage(content=f"Analyze this process step: {step}")
            ])
            row.sources = sources
            row.rpn = row.severity * row.occurrence * row.detection
            row.recommended_action = _suggest_poka_yoke(row)
        except Exception as e:
            print(f"⚠️ FMEA failed for {step}: {e}")
            row = FMEARow(
                process_step=step,
                failure_mode="Process failure",
                effect="Quality defect",
                severity=5,
                cause="Unknown",
                occurrence=5,
                current_control="Inspection",
                detection=5,
                sources=[]
            )
            row.rpn = row.severity * row.occurrence * row.detection
            row.recommended_action = _suggest_poka_yoke(row)
        
        fmea_rows.append(row)
    
    fmea_rows.sort(key=lambda x: x.rpn, reverse=True)
    return {
        "fmea_rows": fmea_rows,
        "top_rpn_risks": fmea_rows[:5], 
        "critical_risks": [r for r in fmea_rows if r.rpn >= 200]
    }

def poka_yoke_node(state: dict) -> dict:
    """Pass-through node since poka-yoke is already applied in fmea_node"""
    return state


def hybrid_fmea_node(state: dict) -> dict:
    """
    Preserves DMAIC state before transitioning to FMEA.
    Ensures selected_path stays as 'hybrid' for downstream processing.
    """
    return {
        "selected_path": "hybrid"
    }
    
def excel_export_node(state: dict) -> dict:
    """Export FMEA or DMAIC results to CSV"""
    fmea_rows = state.get("fmea_rows", [])
    solutions = state.get("solutions", [])
    
    export_data = []
    
    if fmea_rows:
        for row in fmea_rows:
            if hasattr(row, 'model_dump'):
                row_dict = row.model_dump()
            elif hasattr(row, 'dict'):
                row_dict = row.dict()
            else:
                row_dict = row
            export_data.append(row_dict)
        
        df = pd.DataFrame(export_data)
        output_file = "fmea_report.csv"
        df.to_csv(output_file, index=False)
        
        excel_file = "fmea_report.xlsx"
        df.to_excel(excel_file, index=False)
        
        return {
            "excel_report_path": excel_file,
            "export_data": export_data
        }
    
    elif solutions:
        export_data = {
            "problem_statement": state.get("problem_statement", ""),
            "baseline_metric": state.get("baseline_metric", 0),
            "target_metric": state.get("target_metric", 0),
            "sigma_level": state.get("sigma_level", 0),
            "root_causes": ", ".join(state.get("root_causes", [])),
            "cpk_value": state.get("cpk_value", 0)
        }
        
        df = pd.DataFrame([export_data])
        excel_file = "dmaic_report.xlsx"
        df.to_excel(excel_file, index=False)
        
        return {
            "excel_report_path": excel_file,
            "export_data": export_data
        }
    
    return {"excel_report_path": "", "export_data": []}

# ------------RESPONSE CITATIONS-----------    
def response_citations_node(state: dict) -> dict:
    """
    Format and add citations/sources to the response.
    Collects sources from various phases and formats them for display.
    """
    all_sources = []
    
    if state.get("rag_context"):
        all_sources.extend([
            {
                "source": s.get("source", "Unknown"),
                "snippet": s.get("snippet", "")[:200],
                "phase": "Reference"
            } 
            for s in state["rag_context"] 
            if isinstance(s, dict)
        ])
    
    if state.get("fmea_rows"):
        for row in state["fmea_rows"]:
            if hasattr(row, 'sources') and row.sources:
                all_sources.extend([
                    {
                        "source": s.get("source", "Unknown"),
                        "snippet": s.get("snippet", "")[:200],
                        "phase": f"FMEA: {row.process_step}"
                    }
                    for s in row.sources
                ])
    
    seen = set()
    unique_sources = []
    for source in all_sources:
        source_tuple = (source["source"], source["snippet"][:100])
        if source_tuple not in seen:
            seen.add(source_tuple)
            unique_sources.append(source)
    
    citation_text = "📚 **Sources & References:**\n\n"
    if unique_sources:
        for i, source in enumerate(unique_sources[:10], 1):
            citation_text += f"{i}. **{source['source']}** ({source['phase']})\n"
            citation_text += f"   _{source['snippet']}_\n\n"
    else:
        citation_text += "_No external sources were used. Analysis based on AI knowledge._\n"
    
    executive_summary = _generate_executive_summary(state)
    
    return {
        "citations": unique_sources,
        "citation_text": citation_text,
        "executive_summary": executive_summary,
        "total_sources": len(unique_sources)
    }

def _generate_executive_summary(state: dict) -> str:
    """Generate executive summary of the analysis"""
    path = state.get("selected_path", "dmaic").upper()
    problem = state.get("problem_statement", "Not specified")
    
    summary = f"# 📊 {path} Analysis Summary\n\n"
    summary += f"**Problem:** {problem}\n\n"
    summary += f"**Path Selected:** {path}\n\n"
    
    if state.get("baseline_metric"):
        summary += "## 📈 Key Metrics\n"
        summary += f"- Baseline: {state['baseline_metric']}% defects\n"
        summary += f"- Target: {state.get('target_metric', 'N/A')}%\n"
        summary += f"- Sigma Level: {state.get('sigma_level', 'N/A')}\n"
        summary += f"- Cpk Value: {state.get('cpk_value', 'N/A')}\n\n"
    
    if state.get("root_causes"):
        summary += "## 🔍 Top Root Causes\n"
        for i, cause in enumerate(state["root_causes"][:3], 1):
            summary += f"{i}. {cause}\n"
        summary += "\n"
    
    if state.get("solutions"):
        summary += "## 💡 Recommended Solutions\n"
        for i, sol in enumerate(state["solutions"][:3], 1):
            if hasattr(sol, 'description'):
                summary += f"{i}. {sol.description} (Priority: {sol.priority_score})\n"
            elif isinstance(sol, dict):
                summary += f"{i}. {sol.get('description', 'N/A')}\n"
        summary += "\n"
    
    if state.get("critical_risks"):
        summary += "## ⚠️ Critical Risks (RPN ≥ 200)\n"
        for risk in state["critical_risks"][:3]:
            if hasattr(risk, 'process_step'):
                summary += f"- {risk.process_step}: {risk.failure_mode} (RPN: {risk.rpn})\n"
            elif isinstance(risk, dict):
                summary += f"- {risk.get('process_step', 'N/A')}: RPN {risk.get('rpn', 'N/A')}\n"
        summary += "\n"
    
    if state.get("is_sustained") is not None:
        status = "✅" if state["is_sustained"] else "⚠️"
        summary += f"## 🎯 Control Status\n{status} Process is {'sustained' if state['is_sustained'] else 'not yet sustained'}\n"
    
    return summary

# ------------STREAMLIT DISPLAY-----------   
def streamlit_display_node(state: dict) -> dict:
    """
    Prepare all results for Streamlit display.
    Organizes output into a structured format ready for UI rendering.
    """
    display_data = {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "selected_path": state.get("selected_path", "dmaic"),
        "phases_completed": _get_completed_phases(state),
        "summary": state.get("executive_summary", ""),
        "citations": state.get("citations", []),
        "citation_text": state.get("citation_text", ""),
        "excel_report_path": state.get("excel_report_path", ""),
    }
    
    if state.get("selected_path") in ["dmaic", "hybrid"]:
        display_data.update({
            "define": _format_define_results(state),
            "measure": _format_measure_results(state),
            "analyze": _format_analyze_results(state),
            "improve": _format_improve_results(state),
            "control": _format_control_results(state)
        })
    
    if state.get("selected_path") in ["fmea", "hybrid"] or state.get("fmea_rows"):
        display_data["fmea"] = _format_fmea_results(state)
        display_data["critical_risks"] = state.get("critical_risks", [])
        display_data["top_rpn_risks"] = [
            {
                "step": row.process_step if hasattr(row, 'process_step') else row.get('process_step', ''),
                "failure_mode": row.failure_mode if hasattr(row, 'failure_mode') else row.get('failure_mode', ''),
                "rpn": row.rpn if hasattr(row, 'rpn') else row.get('rpn', 0),
                "recommended_action": row.recommended_action if hasattr(row, 'recommended_action') else row.get('recommended_action', '')
            }
            for row in state.get("top_rpn_risks", [])[:5]
        ]
    
    display_data["formatted_output"] = _generate_formatted_output(display_data)
    
    logger.info(f"✅ Streamlit display prepared: {len(display_data.get('formatted_output', ''))} chars")
    
    return {
        "display_data": display_data,
        "ready_for_display": True
    }

def _get_completed_phases(state: dict) -> list:
    """Determine which phases were completed"""
    phases = []
    
    if state.get("selected_path") in ["dmaic", "hybrid"]:
        if state.get("project_charter") or state.get("ctq_list"):
            phases.append("Define")
        if state.get("baseline_metric"):
            phases.append("Measure")
        if state.get("root_causes"):
            phases.append("Analyze")
        if state.get("solutions"):
            phases.append("Improve")
        if state.get("control_plan"):
            phases.append("Control")
    
    if state.get("fmea_rows"):
        phases.append("FMEA")
    
    return phases

def _format_define_results(state: dict) -> dict:
    """Format Define phase results for display"""
    charter = state.get("project_charter")
    return {
        "problem_statement": state.get("problem_statement", ""),
        "charter": charter.model_dump() if hasattr(charter, 'model_dump') else charter,
        "ctq_list": state.get("ctq_list", []),
        "sipoc": state.get("sipoc", {}),
        "swot": state.get("swot", {})
    }

def _format_measure_results(state: dict) -> dict:
    """Format Measure phase results for display"""
    return {
        "baseline_metric": state.get("baseline_metric", 0),
        "target_metric": state.get("target_metric", 0),
        "sigma_level": state.get("sigma_level", 0),
        "defects_per_unit": state.get("defects_per_unit", 0),
        "data_collection_plan": state.get("data_collection_plan", ""),
        "improvement_gap": (state.get("baseline_metric", 0) - state.get("target_metric", 0))
    }

def _format_analyze_results(state: dict) -> dict:
    """Format Analyze phase results for display"""
    return {
        "root_causes": state.get("root_causes", []),
        "pareto_top_causes": state.get("pareto_top_causes", []),
        "fishbone_diagram": state.get("fishbone_json", {}),
        "five_whys": state.get("root_causes", [])
    }

def _format_improve_results(state: dict) -> dict:
    """Format Improve phase results for display"""
    solutions = state.get("solutions", [])
    formatted_solutions = []
    
    for sol in solutions:
        if hasattr(sol, 'model_dump'):
            formatted_solutions.append(sol.model_dump())
        elif hasattr(sol, 'dict'):
            formatted_solutions.append(sol.dict())
        else:
            formatted_solutions.append(sol)
    
    selected = state.get("selected_solution")
    return {
        "solutions": formatted_solutions,
        "selected_solution": selected.model_dump() if hasattr(selected, 'model_dump') else selected,
        "total_solutions": len(formatted_solutions)
    }

def _format_control_results(state: dict) -> dict:
    """Format Control phase results for display"""
    return {
        "cpk_value": state.get("cpk_value", 0),
        "spc_limits": state.get("spc_limits", (0, 0, 0)),
        "control_plan": state.get("control_plan", ""),
        "is_sustained": state.get("is_sustained", False),
        "cpk_interpretation": _interpret_cpk(state.get("cpk_value", 0))
    }

def _format_fmea_results(state: dict) -> dict:
    """Format FMEA results for display"""
    fmea_rows = state.get("fmea_rows", [])
    formatted_rows = []
    
    for row in fmea_rows:
        if hasattr(row, 'model_dump'):
            formatted_rows.append(row.model_dump())
        elif hasattr(row, 'dict'):
            formatted_rows.append(row.dict())
        else:
            formatted_rows.append(row)
    
    return {
        "fmea_rows": formatted_rows,
        "total_steps": len(formatted_rows),
        "average_rpn": sum(r.get('rpn', 0) for r in formatted_rows) / len(formatted_rows) if formatted_rows else 0,
        "high_risk_count": len([r for r in formatted_rows if r.get('rpn', 0) >= 200])
    }

def _interpret_cpk(cpk: float) -> str:
    if cpk is None:
        return "⚪ Insufficient data"
    if cpk >= 1.33:
        return "✅ Excellent - Process is capable"
    elif cpk >= 1.0:
        return "⚠️ Adequate - Process is marginally capable"
    elif cpk >= 0.67:
        return "🟠 Poor - Process needs improvement"
    else:
        return "🔴 Critical - Process is not capable"

def _generate_formatted_output(display_data: dict) -> str:
    """Generate formatted markdown output for Streamlit"""
    output = []
    
    output.append(f"# 🎯 {display_data.get('selected_path', 'DMAIC').upper()} Analysis Report")
    output.append(f"*Generated: {display_data.get('timestamp', 'N/A')}*\n")
    
    if display_data.get("summary"):
        output.append(display_data["summary"])
    
    if "define" in display_data:
        output.append("## 📋 Define Phase\n")
        define = display_data["define"]
        output.append(f"**Problem Statement:** {define.get('problem_statement', 'N/A')}\n")
        if define.get("ctq_list"):
            output.append("**CTQs:**")
            for ctq in define["ctq_list"]:
                output.append(f"- {ctq}")
        output.append("")
    
    if "measure" in display_data:
        output.append("## 📏 Measure Phase\n")
        measure = display_data["measure"]
        output.append(f"- Baseline: {measure.get('baseline_metric', 0)}%")
        output.append(f"- Target: {measure.get('target_metric', 0)}%")
        output.append(f"- Sigma Level: {measure.get('sigma_level', 0)}")
        output.append(f"- Improvement Gap: {measure.get('improvement_gap', 0)}%\n")
    
    if "analyze" in display_data:
        output.append("## 🔍 Analyze Phase\n")
        analyze = display_data["analyze"]
        if analyze.get("root_causes"):
            output.append("**Root Causes:**")
            for cause in analyze["root_causes"][:5]:
                output.append(f"- {cause}")
        output.append("")
    
    if "improve" in display_data:
        output.append("## 💡 Improve Phase\n")
        improve = display_data["improve"]
        if improve.get("solutions"):
            output.append("**Proposed Solutions:**")
            for sol in improve["solutions"][:3]:
                output.append(f"- {sol.get('description', 'N/A')}")
                if sol.get('priority_score'):
                    output.append(f"  Priority Score: {sol['priority_score']}")
        output.append("")
    
    if "control" in display_data:
        output.append("## 🎛️ Control Phase\n")
        control = display_data["control"]
        output.append(f"- Cpk: {control.get('cpk_value', 0)} ({control.get('cpk_interpretation', 'N/A')})")
        output.append(f"- SPC Limits: {control.get('spc_limits', (0,0,0))}")
        output.append(f"- Sustained: {'Yes' if control.get('is_sustained') else 'No'}\n")
    
    if "fmea" in display_data:
        output.append("## ⚡ FMEA Analysis\n")
        fmea = display_data["fmea"]
        output.append(f"- Total Steps Analyzed: {fmea.get('total_steps', 0)}")
        output.append(f"- Average RPN: {fmea.get('average_rpn', 0):.1f}")
        output.append(f"- High Risk Items: {fmea.get('high_risk_count', 0)}\n")
        
        if display_data.get("top_rpn_risks"):
            output.append("**Top Risks:**")
            for risk in display_data["top_rpn_risks"][:3]:
                output.append(f"- {risk.get('step', 'N/A')}: {risk.get('failure_mode', 'N/A')} (RPN: {risk.get('rpn', 0)})")
        output.append("")
    
    if display_data.get("citation_text"):
        output.append(display_data["citation_text"])
    
    return "\n".join(output)




# 9. BUILD GRAPH
builder = StateGraph(DMAICState)

# Add all nodes
builder.add_node("collect_user_input", collect_user_input_node)
builder.add_node("langsmith_trace", langsmith_trace_node)
builder.add_node("router", router_node)
builder.add_node("rag_retrieve_dmaic", rag_retrieve_dmaic_node)
builder.add_node("rag_retrieve_fmea", rag_retrieve_fmea_node)
builder.add_node("rag_retrieve_hybrid", rag_retrieve_hybrid_node)
builder.add_node("vector_store", vector_store_node)
builder.add_node("define", define_node)
builder.add_node("measure", measure_node)
builder.add_node("analyze", analyze_node)
builder.add_node("improve", improve_node)
builder.add_node("control", control_node)
builder.add_node("fmea", fmea_node)
builder.add_node("poka_yoke", poka_yoke_node)
builder.add_node("hybrid_fmea", hybrid_fmea_node)
builder.add_node("excel_export", excel_export_node)
builder.add_node("response_citations", response_citations_node)
builder.add_node("streamlit_display", streamlit_display_node)

# Set entry point
builder.set_entry_point("collect_user_input")

# Initial edges
builder.add_edge("collect_user_input", "langsmith_trace")
builder.add_edge("langsmith_trace", "router")

# Router to RAG
builder.add_conditional_edges(
    "router",
    lambda state: state.get("selected_path", "dmaic"),
    {
        "dmaic": "rag_retrieve_dmaic",
        "fmea": "rag_retrieve_fmea",
        "hybrid": "rag_retrieve_hybrid"
    }
)

# RAG to vector_store
builder.add_edge("rag_retrieve_dmaic", "vector_store")
builder.add_edge("rag_retrieve_fmea", "vector_store")
builder.add_edge("rag_retrieve_hybrid", "vector_store")

# vector_store routes to correct path
builder.add_conditional_edges(
    "vector_store",
    lambda state: state.get("selected_path", "dmaic", ),
    {
        "dmaic": "define",
        "fmea": "fmea",
        "hybrid": "define"
    }
)

# === DMAIC PATH ===
builder.add_edge("define", "measure")
builder.add_edge("measure", "analyze")
builder.add_edge("analyze", "improve")
builder.add_edge("improve", "control")

# control splits: DMAIC goes to excel, hybrid goes to hybrid_preserve
builder.add_conditional_edges(
    "control",
    lambda state: state.get("selected_path", "dmaic"),
    {
        "dmaic": "excel_export",
        "hybrid": "hybrid_fmea"
    }
)
# === HYBRID PATH === (continues after DMAIC)
builder.add_edge("hybrid_fmea", "fmea")

# === FMEA PATH ===
builder.add_edge("fmea", "poka_yoke")
builder.add_edge("poka_yoke", "excel_export")

# Final output
builder.add_edge("excel_export", "response_citations")
builder.add_edge("response_citations", "streamlit_display")
builder.add_edge("streamlit_display", END)

# Compile with checkpointer
graph = builder.compile()
graph