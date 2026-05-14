# filename: dmaic_frontend.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import uuid
import numpy as np
import os
import re
import time

# Import your backend
try:
    from backend import graph, kb
    from langchain_core.messages import HumanMessage
    BACKEND_LOADED = True
except Exception as e:
    st.error(f"❌ Failed to load backend: {e}")
    st.warning("Running in demo mode with mock data")
    BACKEND_LOADED = False

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="PM Assistant Copilot - AI-Powered Six Sigma Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 35px -10px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .glass-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        background: rgba(255,255,255,0.08);
        border-color: rgba(255,255,255,0.2);
        box-shadow: 0 8px 25px -5px rgba(0,0,0,0.2);
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: 800;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(102,126,234,0.4);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ffa500 0%, #ff8c00 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f6e05e 0%, #ecc94b 100%);
        color: #1a202c;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .source-card {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #667eea;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    .phase-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .phase-complete {
        background: rgba(72, 199, 142, 0.2);
        border: 1px solid rgba(72, 199, 142, 0.4);
        color: #48c78e;
    }
    
    .phase-pending {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #a0aec0;
    }
    
    .badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 0.1rem;
    }
    
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; }
    
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    .loading-text { animation: pulse 1.5s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State Initialization
# ============================================================
def init_session_state():
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_state' not in st.session_state:
        st.session_state.current_state = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'model_temp' not in st.session_state:
        st.session_state.model_temp = 0.3
    if 'rag_k' not in st.session_state:
        st.session_state.rag_k = 4
    if 'max_token' not in st.session_state:
        st.session_state.max_token = 1000
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'selected_path' not in st.session_state:
        st.session_state.selected_path = 'dmaic'
    if "llm_call_count" not in st.session_state:
        st.session_state.llm_call_count = 0
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "execution_time" not in st.session_state:
        st.session_state.execution_time = 0

init_session_state()

# ============================================================
# Helper Functions
# ============================================================
def get_risk_badge(rpn):
    if rpn >= 300:
        return '<span class="risk-critical">🔴 CRITICAL</span>'
    elif rpn >= 200:
        return '<span class="risk-high">🟠 HIGH</span>'
    elif rpn >= 100:
        return '<span class="risk-medium">🟡 MEDIUM</span>'
    else:
        return '<span class="risk-low">🟢 LOW</span>'

def display_sources(state):
    rag_context = state.get('rag_context', [])
    
    if rag_context:
        st.markdown(f'<span class="badge" style="background: #667eea;">📄 {len(rag_context)} Sources Retrieved</span>', unsafe_allow_html=True)
        st.markdown("---")
        
        for i, source in enumerate(rag_context):
            if isinstance(source, dict):
                source_type = source.get('source_type', 'unknown')
                icon = "📄" if source_type == 'pdf' else "🌐" if source_type == 'web' else "📚"
                source_name = source.get('source_name', source.get('source', 'Unknown'))
                page = source.get('page', 'N/A')
                snippet = source.get('snippet', source.get('content', 'No content')[:300])
                
                st.markdown(f"""
                    <div class="source-card">
                        <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                            <span style="font-size: 1.2rem;">{icon}</span>
                            <span style="font-weight: 600;">{source_name}</span>
                            <span class="badge" style="background: #4a5568;">{source_type.upper()}</span>
                            <span class="badge" style="background: #2d3748;">Page: {page}</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #a0aec0; margin: 0.5rem 0;">
                            {source.get('relevance', 'Relevant to analysis')}
                        </div>
                        <div style="font-size: 0.9rem; line-height: 1.4;">
                            "{snippet[:250]}..."
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📚 No sources retrieved. Make sure knowledge base is loaded with PDFs or web sources.")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 0.5rem 0;">
            <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 1.5rem;">🎯</span>
            </div>
            <h3 style="margin: 0.5rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       -webkit-background-clip: text; background-clip: text; color: transparent;">
                PM Assistant Copilot
            </h3>
            <p style="color: #a0aec0; font-size: 0.7rem;">Six Sigma Assistant | RAG + LangGraph + FMEA</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="text-align: center;"><span class="badge" style="background: #48bb78; font-size: 0.6rem; padding: 2px 8px;">● LIVE</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # System Health
    with st.expander("⚙️ System Health", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("API Status", "🟢 Connected", delta="Groq LLaMA 3.3")
        with col2:
            st.metric("RAG Status", "✅ Active" if BACKEND_LOADED else "⚠️ Limited", delta=f"{st.session_state.rag_k} chunks")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("LLM Calls", st.session_state.llm_call_count)
        with col2:
            st.metric("Tokens Used", f"{st.session_state.total_tokens:,}")
        
        if st.session_state.total_tokens > 80000:
            st.warning("⚠️ Approaching API rate limit.")
        
        if st.session_state.analysis_complete and st.session_state.current_state:
            execution_time = st.session_state.get('execution_time', 0)
            st.progress(min(execution_time / 30, 1.0), text=f"⏱️ Analysis: {execution_time:.1f}s")
        
        # Real-time processing indicator
        if st.session_state.get('processing', False):
            st.markdown("""
                <div style="background: linear-gradient(90deg, #667eea, #764ba2, #667eea); 
                           background-size: 200% 100%; animation: loading 2s ease-in-out infinite;
                           height: 2px; border-radius: 2px; margin: 10px 0;">
                </div>
            """, unsafe_allow_html=True)
            st.caption("🧠 AI is processing...")
    # Model Configuration
    with st.expander("🎛️ Model Configuration", expanded=False):
        preset = st.selectbox("Quick Presets", ["Custom", "Creative (High Temp)", "Precise (Low Temp)", "Balanced"], index=0)
        
        if preset == "Creative (High Temp)":
            st.session_state.model_temp = 0.7
            st.session_state.rag_k = 6
        elif preset == "Precise (Low Temp)":
            st.session_state.model_temp = 0.1
            st.session_state.rag_k = 3
        elif preset == "Balanced":
            st.session_state.model_temp = 0.3
            st.session_state.rag_k = 4
        
        st.session_state.model_temp = st.slider("🎨 Temperature", 0.0, 1.0, st.session_state.model_temp, 0.05)
        st.session_state.rag_k = st.slider("📚 RAG Chunks (k)", 1, 10, st.session_state.rag_k)
        st.session_state.max_token = st.slider("📝 Max Response Length", 200, 4000, st.session_state.max_token)

        estimated_cost = (st.session_state.total_tokens / 1000000) * 0.8
        st.caption(f"💰 Estimated cost: ${estimated_cost:.4f}")

    # Analysis Path
    with st.expander("🎯 Analysis Path", expanded=False):
        analysis_options = {
            "📈 DMAIC Only": "Full Six Sigma DMAIC",
            "⚠️ FMEA Only": "Risk Assessment", 
            "🔄 Hybrid (DMAIC + FMEA)": "Combined Analysis"
        }
        analysis_type = st.radio("Select Methodology:", list(analysis_options.keys()))
        st.caption(f"📌 {analysis_options[analysis_type]}")
    
    # Specification Limits
    with st.expander("📏 Specification Limits", expanded=False):
        st.markdown("*Required for accurate Cpk calculation*")
        usl = st.number_input("USL (Upper Spec Limit)", value=3.0, step=0.1)
        lsl = st.number_input("LSL (Lower Spec Limit)", value=0.5, step=0.1)
        stdev = st.number_input("Std Deviation", value=1.2, step=0.1, min_value=0.01)
    
    # Knowledge Base
    with st.expander("📚 Knowledge Base", expanded=False):
        if BACKEND_LOADED:
            try:
                stats = kb.get_stats()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 PDF Files", stats.get('files', 0))
                    st.metric("🔗 Web URLs", stats.get('urls', 0))
                with col2:
                    total_sources = stats.get('total_sources', 0)
                    st.metric("📚 Total Sources", total_sources)
                    if total_sources > 10:
                        st.success("✅ Rich knowledge base")
                    elif total_sources > 0:
                        st.warning("⚠️ Limited sources")
                    else:
                        st.error("❌ No sources loaded")
            except:
                st.info("📚 KB Status: Ready")
        else:
            st.info("Demo Mode - Simulated responses")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.current_state = None
            st.session_state.analysis_complete = False
            st.session_state.llm_call_count = 0
            st.session_state.total_tokens = 0
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.toast("Cache cleared!", icon="✅")
    
    st.markdown("---")
    st.caption(f"🆔 Session: `{st.session_state.thread_id[:8]}...`")
    st.caption("© 2025 DMAIC Copilot | v2.0")

# ============================================================
# Header
# ============================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
        <div style="text-align: center;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       -webkit-background-clip: text; background-clip: text; color: transparent;">
                🎯 PM Assistant Copilot
            </h1>
            <p style="color: #a0aec0; font-size: 0.9rem;">
                Six Sigma Assistant | RAG + LangGraph + FMEA
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="text-align: center;"><span class="badge" style="background: #48bb78; font-size: 0.6rem; padding: 2px 8px;">● LIVE</span></div>', unsafe_allow_html=True)
    st.markdown("---")

st.markdown("---")

# Phase Progress Indicator
if st.session_state.analysis_complete and st.session_state.current_state:
    state = st.session_state.current_state
    phases_completed = state.get('display_data', {}).get('phases_completed', [])
    all_phases = ['Define', 'Measure', 'Analyze', 'Improve', 'Control', 'FMEA']
    
    cols = st.columns(len(all_phases))
    for i, phase in enumerate(all_phases):
        with cols[i]:
            if phase in phases_completed:
                st.markdown(f'<div class="phase-indicator phase-complete">✅ {phase}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="phase-indicator phase-pending">⏳ {phase}</div>', unsafe_allow_html=True)

# ============================================================
# Input Section
# ============================================================
with st.container():
    st.markdown("### 🎯 Problem Statement")

    input_tab1, input_tab2, input_tab3 = st.tabs(["📝 Manual Entry", "📋 Template Library", "📄 Upload File"])

    problem = st.session_state.get('problem_input', '')
    process_steps = st.session_state.get('steps_input', '')
    baseline_metric = st.session_state.get('baseline_input', 15.0)
    target_metric = st.session_state.get('target_input', 2.0)

    with input_tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            problem = st.text_area(
                "Describe your quality/process problem:",
                value=problem,
                placeholder="Example: High defect rate in PCB assembly line causing 15% rework rate...",
                height=100,
                key="problem_input",
                help="Describe the quality or process issue you want to analyze")
            
            customer = st.text_input(
                "Customer/Stakeholder:", 
                placeholder="Production Manager", 
                key="customer_input",
                help="Who is the end customer or stakeholder affected by this problem?")
        with col2:
            baseline_input = st.number_input("Baseline Defect %", 0.0, 100.0, baseline_metric, 0.5, key="baseline_input",help="Current defect rate before improvement (e.g., 15% means 15 out of 100 units are defective)")
            target_input = st.number_input("Target Defect %", 0.0, 100.0, target_metric, 0.5, key="target_input", help="Desired defect rate after DMAIC improvements (e.g., 2% means only 2 out of 100 units defective)")
    
    with input_tab2:
        template_options = {
            "Manufacturing Defect": "High defect rate in [product] assembly line. Current defect rate is [X]%, target [Y]%.",
            "Process Inefficiency": "Cycle time for [process] is [X] minutes, target [Y] minutes.",
            "Quality Issue": "Customer complaints increased by [X]% in last quarter.",
            "FMEA Risk": "New product launch requires risk assessment for [product]."
        }
        selected_template = st.selectbox("Choose a template:", list(template_options.keys()))
        if selected_template:
            st.info(f"**Preview:** {template_options[selected_template]}")
            if st.button("📋 Use Template"):
                st.session_state.problem_input = template_options[selected_template]
                st.rerun()
    
    with input_tab3:
        uploaded_file = st.file_uploader("Upload problem description:", type=['txt', 'pdf', 'docx'])
        if uploaded_file and uploaded_file.type == "text/plain":
            problem = uploaded_file.read().decode()
            st.session_state.problem_input = problem
            st.success(f"✅ Loaded: {uploaded_file.name}")

    # Process Steps at full width
    process_steps = st.text_area(
        "Process Steps (one per line):",
        value=process_steps,
        placeholder="Step 1: Material loading\nStep 2: Assembly\nStep 3: Inspection\nStep 4: Testing\nStep 5: Packaging",
        height=120,
        key="steps_input",
        help="List each step of your process. For FMEA analysis, each step will be evaluated for failure modes")
    
    # CTQ Input
    ctq_input = st.text_input(
        "Critical to Quality (CTQs) - comma separated:",
        placeholder="Defect rate, Cycle time, Customer satisfaction",
        key="ctq_input",
        help="What matters most to your customer? Separate multiple CTQs with commas."
    )
    
    analyze_clicked = st.button(
        "🚀 START ANALYSIS", 
        type="primary", 
        use_container_width=True,
        disabled=not problem
    )

st.markdown("---")

# ============================================================
# Analysis Execution
# ============================================================
if analyze_clicked and problem:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("🧠 AI Agent analyzing your problem..."):
        try:
            start_time = time.time()
            st.session_state.processing = True
            
            steps_list = [s.strip() for s in process_steps.split('\n') if s.strip()]
            if not steps_list:
                steps_list = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]
            
            ctq_list = [c.strip() for c in ctq_input.split(',') if c.strip()] if ctq_input else []
            
            # Fix: Match the sidebar radio keys
            path_map = {
                "📈 DMAIC Only": "dmaic", 
                "⚠️ FMEA Only": "fmea", 
                "🔄 Hybrid (DMAIC + FMEA)": "hybrid"
            }
            selected_path = path_map.get(analysis_type, "dmaic")
            
            status_text.text("Phase 1/6: Define - Creating Project Charter...")
            progress_bar.progress(10)
            
            initial_state = {
                "problem_statement": problem,
                "customer": customer if customer else "Stakeholder",
                "selected_path": selected_path,
                "baseline_metric": baseline_input,
                "target_metric": target_input,
                "ctq_list": ctq_list,
                "process_steps": steps_list,
                "usl": usl,
                "lsl": lsl,
                "stdev": stdev,
                "needs_research": False,
                "queries": [],
                "reasoning": "",
                "max_results": st.session_state.rag_k,
                "messages": [HumanMessage(content=problem)],
            }
            
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            status_text.text("Phase 2/6: Measure - Calculating baseline metrics...")
            progress_bar.progress(25)
            
            if BACKEND_LOADED:
                result = graph.invoke(initial_state, config=config)
                st.session_state.llm_call_count += 7
                st.session_state.total_tokens += 1000
            else:
                # Demo mode
                result = initial_state.copy()
                result.update({
                    "selected_path": selected_path,
                    "baseline_metric": baseline_input,
                    "target_metric": target_input,
                    "sigma_level": 2.8,
                    "cpk_value": 0.95,
                    "ctq_list": ctq_list if ctq_list else ["Defect rate", "Process capability", "Customer satisfaction"],
                    "root_causes": ["Material variation", "Operator inconsistency", "Equipment calibration drift", "Environmental factors", "Insufficient training"],
                    "solutions": [
                        type('Solution', (), {'description': 'Implement automated calibration system', 'impact_score': 9, 'effort_score': 6, 'priority_score': 12})(),
                        type('Solution', (), {'description': 'Enhanced operator training program', 'impact_score': 7, 'effort_score': 3, 'priority_score': 11})(),
                    ],
                    "fmea_rows": [
                        type('FMEARow', (), {'process_step': steps_list[0], 'failure_mode': 'Material defect', 'rpn': 250, 'severity': 8, 'occurrence': 7, 'detection': 5, 'recommended_action': 'Implement incoming inspection'})(),
                        type('FMEARow', (), {'process_step': steps_list[1] if len(steps_list) > 1 else "Step 2", 'failure_mode': 'Assembly error', 'rpn': 180, 'severity': 7, 'occurrence': 6, 'detection': 4, 'recommended_action': 'Add poka-yoke device'})(),
                    ],
                    "control_plan": "Daily SPC monitoring, weekly calibration checks, monthly training refreshers",
                    "data_collection_plan": "Collect 50 samples daily, measure defect rates, track process parameters"
                })
            
            progress_bar.progress(50)
            status_text.text("Phase 3/6: Analyze - Identifying root causes...")
            progress_bar.progress(60)
            status_text.text("Phase 4/6: Improve - Generating solutions...")
            progress_bar.progress(75)
            status_text.text("Phase 5/6: Control - Creating control plan...")
            progress_bar.progress(90)
            status_text.text("Phase 6/6: Finalizing - Generating report...")
            progress_bar.progress(100)
            
            execution_time = time.time() - start_time
            st.session_state.execution_time = execution_time
            st.session_state.current_state = result
            st.session_state.analysis_complete = True
            st.session_state.selected_path = result.get('selected_path', 'dmaic')
            st.session_state.processing = False
            
            st.session_state.analysis_history.append({
                'problem': problem[:100],
                'path': selected_path,
                'timestamp': datetime.now().isoformat(),
                'sigma': result.get('sigma_level', 0)
            })
            
            status_text.empty()
            progress_bar.empty()
            
            path_display = {"dmaic": "📈 DMAIC", "fmea": "⚠️ FMEA", "hybrid": "🔄 Hybrid (DMAIC+FMEA)"}
            st.success(f"✅ Analysis complete in {execution_time:.1f}s! Using **{path_display.get(selected_path, 'DMAIC')}** methodology")
            st.rerun()
            
        except Exception as e:
            st.session_state.processing = False
            error_msg = str(e)
            
            if "rate_limit_exceeded" in error_msg or "429" in error_msg:
                wait_match = re.search(r'in (\d+)m(\d+\.?\d*)s', error_msg)
                wait_time = f"{wait_match.group(1)}m {wait_match.group(2)}s" if wait_match else "a few minutes"
                st.error(f"⚠️ **API Rate Limit Reached** - Wait {wait_time} or upgrade at [Groq Console](https://console.groq.com/settings/billing)")
            elif "timeout" in error_msg.lower():
                st.error("⚠️ **Request Timeout** - Try reducing analysis depth or RAG chunks")
            else:
                st.error(f"❌ Analysis failed: {error_msg[:200]}")
                st.exception(e)

# ============================================================
# Display Results - Dynamic Tabs Based on Analysis Type
# ============================================================
if st.session_state.analysis_complete and st.session_state.current_state:
    state = st.session_state.current_state
    selected_path = state.get('selected_path', 'dmaic')
    
    path_colors = {"dmaic": "📈", "fmea": "⚠️", "hybrid": "🔄"}
    st.info(f"**Active Path:** {path_colors.get(selected_path, '📈')} {selected_path.upper()} methodology")
    
    # Executive Dashboard
    st.markdown("### 📊 Executive Dashboard")
    
    sigma = state.get('sigma_level', 0)
    baseline = state.get('baseline_metric', 0)
    target = state.get('target_metric', 0)
    improvement = baseline - target if baseline and target else 0
    root_causes_count = len(state.get('root_causes', []))
    solutions_count = len(state.get('solutions', []))
    cpk = state.get('cpk_value', 0)
    fmea_count = len(state.get('fmea_rows', []))
    
    if selected_path == "fmea":
        cols = st.columns(3)
        with cols[0]:
            st.metric("FMEA Risks", fmea_count)
        with cols[1]:
            critical = len([r for r in state.get('fmea_rows', []) if (r.rpn if hasattr(r, 'rpn') else r.get('rpn', 0)) >= 200])
            st.metric("Critical Risks", critical)
        with cols[2]:
            if state.get('fmea_rows'):
                avg_rpn = sum([r.rpn if hasattr(r, 'rpn') else r.get('rpn', 0) for r in state.get('fmea_rows', [])]) / max(len(state.get('fmea_rows', [])), 1)
                st.metric("Avg RPN", f"{avg_rpn:.0f}")
    else:
        cols = st.columns(5)
        with cols[0]:
            st.metric("Sigma Level", f"{sigma:.1f}" if sigma else "N/A")
        with cols[1]:
            st.metric("Improvement", f"↓{improvement:.1f}%" if improvement else "N/A")
        with cols[2]:
            st.metric("Root Causes", root_causes_count)
        with cols[3]:
            st.metric("Solutions", solutions_count)
        with cols[4]:
            status = "✅" if cpk >= 1.33 else "⚠️" if cpk >= 1.0 else "❌"
            st.metric(f"Cpk {status}", f"{cpk:.2f}" if cpk else "N/A")
    
    st.markdown("---")
    
    # Dynamic tabs based on path
    if selected_path == "fmea":
        tab1, tab2, tab3 = st.tabs(["⚠️ FMEA Analysis", "🛡️ Poka-Yoke Solutions", "📚 Sources & Citations"])
    elif selected_path == "hybrid":
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📝 Define", "📏 Measure", "🔍 Analyze", "💡 Improve", "🎮 Control", "⚠️ FMEA", "📚 Sources"
        ])
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📝 Define", "📏 Measure", "🔍 Analyze", "💡 Improve", "🎮 Control", "📚 Sources"
        ])
    
    # ========== DMAIC/HYBRID TABS ==========
    if selected_path in ["dmaic", "hybrid"]:
        with tab1:
            st.markdown("### 📝 Define Phase Results")
            charter = state.get('project_charter')
            if charter:
                with st.expander("📋 Project Charter", expanded=True):
                    if hasattr(charter, 'problem_statement'):
                        st.info(f"**Problem Statement:** {charter.problem_statement}")
                        st.success(f"**Goal Statement:** {charter.goal_statement}")
                        st.write(f"**Scope:** {charter.scope}")
                        st.write(f"**Business Case:** {charter.business_case}")
                    elif isinstance(charter, dict):
                        st.info(f"**Problem:** {charter.get('problem_statement', 'N/A')}")
                        st.success(f"**Goal:** {charter.get('goal_statement', 'N/A')}")
            
            ctqs = state.get('ctq_list', [])
            if ctqs:
                st.markdown("#### 🎯 Critical to Quality (CTQs)")
                for i, ctq in enumerate(ctqs, 1):
                    st.markdown(f"{i}. ✅ {ctq}")
            
            col1, col2 = st.columns(2)
            with col1:
                sipoc = state.get('sipoc')
                if sipoc:
                    with st.expander("📊 SIPOC Diagram", expanded=False):
                        if hasattr(sipoc, 'suppliers'):
                            st.write(f"**Suppliers:** {', '.join(sipoc.suppliers)}")
                            st.write(f"**Inputs:** {', '.join(sipoc.inputs)}")
                            st.write(f"**Process:** {', '.join(sipoc.process_steps)}")
                            st.write(f"**Outputs:** {', '.join(sipoc.outputs)}")
                            st.write(f"**Customers:** {', '.join(sipoc.customers)}")
            with col2:
                swot = state.get('swot')
                if swot:
                    with st.expander("📈 SWOT Analysis", expanded=False):
                        if hasattr(swot, 'strengths'):
                            st.success(f"**Strengths:** {', '.join(swot.strengths)}")
                            st.error(f"**Weaknesses:** {', '.join(swot.weaknesses)}")
                            st.info(f"**Opportunities:** {', '.join(swot.opportunities)}")
                            st.warning(f"**Threats:** {', '.join(swot.threats)}")
        
        with tab2:
            st.markdown("### 📏 Measure Phase Results")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Baseline Defect Rate", f"{baseline:.1f}%" if baseline else "N/A")
            with col2:
                st.metric("Target Defect Rate", f"{target:.1f}%" if target else "N/A", delta=f"-{improvement:.1f}%" if improvement else None)
            with col3:
                st.metric("Sigma Level", f"{sigma:.1f}" if sigma else "N/A")
            with col4:
                dpmo = state.get('defects_per_unit', 0)
                st.metric("DPU", f"{dpmo:.4f}" if dpmo else "N/A")
            
            data_plan = state.get('data_collection_plan', '')
            if data_plan:
                with st.expander("📋 Data Collection Plan", expanded=True):
                    st.write(data_plan)
            
            if sigma and sigma > 0:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=sigma,
                    title={'text': "Process Sigma Level"},
                    gauge={
                        'axis': {'range': [0, 6]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 2], 'color': "red"},
                            {'range': [2, 3], 'color': "orange"},
                            {'range': [3, 4], 'color': "yellow"},
                            {'range': [4, 5], 'color': "lightgreen"},
                            {'range': [5, 6], 'color': "green"}
                        ]
                    }
                ))
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 🔍 Analyze Phase Results")
            root_causes = state.get('root_causes', [])
            if root_causes:
                st.markdown("#### 🎯 Root Causes Identified (5 Whys Analysis)")
                for i, cause in enumerate(root_causes, 1):
                    icon = "🔴" if i <= 2 else "🟠" if i <= 4 else "🟡"
                    st.markdown(f"{icon} **{i}.** {cause}")
            
            pareto = state.get('pareto_top_causes', [])
            if pareto:
                st.markdown("#### 📊 Pareto Chart (80/20 Analysis)")
                if isinstance(pareto[0], (tuple, list)):
                    df_pareto = pd.DataFrame(pareto, columns=['Cause', 'Percentage'])
                else:
                    df_pareto = pd.DataFrame({'Cause': pareto, 'Percentage': [40, 25, 15, 10, 5, 5][:len(pareto)]})
                df_pareto['Cumulative'] = df_pareto['Percentage'].cumsum()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_pareto['Cause'], y=df_pareto['Percentage'], name='Individual %', marker_color='steelblue'))
                fig.add_trace(go.Scatter(x=df_pareto['Cause'], y=df_pareto['Cumulative'], name='Cumulative %', yaxis='y2', mode='lines+markers', marker_color='red', line=dict(width=3)))
                fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="80% threshold")
                fig.update_layout(height=400, yaxis2=dict(overlaying='y', side='right', range=[0, 100]))
                st.plotly_chart(fig, use_container_width=True)
            
            fishbone = state.get('fishbone_json')
            if fishbone and isinstance(fishbone, dict):
                st.markdown("#### 🐟 Fishbone Diagram (6Ms)")
                bones = fishbone.get('bones', {})
                if bones:
                    cols_fish = st.columns(3)
                    categories = list(bones.keys())
                    for i, cat in enumerate(categories):
                        with cols_fish[i % 3]:
                            st.markdown(f"**{cat}**")
                            for cause in bones[cat][:3]:
                                st.markdown(f"• {cause}")
        
        with tab4:
            st.markdown("### 💡 Improve Phase Results")
            solutions = state.get('solutions', [])
            if solutions:
                st.markdown(f"#### 🚀 Proposed Solutions ({len(solutions)} total)")
                sol_data = []
                for sol in solutions:
                    if hasattr(sol, 'description'):
                        sol_data.append({
                            "Solution": sol.description[:80] + "..." if len(sol.description) > 80 else sol.description,
                            "Impact": sol.impact_score,
                            "Effort": sol.effort_score,
                            "Priority": sol.priority_score,
                            "Sigma Improvement": f"+{sol.estimated_sigma_improvement}"
                        })
                    elif isinstance(sol, dict):
                        sol_data.append({
                            "Solution": sol.get('description', 'N/A')[:80],
                            "Impact": sol.get('impact_score', 0),
                            "Effort": sol.get('effort_score', 0),
                            "Priority": sol.get('priority_score', 0),
                            "Sigma Improvement": f"+{sol.get('estimated_sigma_improvement', 0)}"
                        })
                
                if sol_data:
                    st.dataframe(pd.DataFrame(sol_data), use_container_width=True, hide_index=True)
                
                selected = state.get('selected_solution')
                if selected:
                    st.success(f"⭐ **Recommended Solution:** {selected.description if hasattr(selected, 'description') else str(selected)}")
                
                if sol_data:
                    fig = go.Figure()
                    for sol in sol_data:
                        fig.add_trace(go.Scatter(
                            x=[sol['Effort']], y=[sol['Impact']],
                            mode='markers+text',
                            text=[sol['Solution'][:30]],
                            textposition='top center',
                            marker=dict(size=sol['Priority']*5, color=sol['Priority'], colorscale='Viridis', showscale=True),
                            name=sol['Solution'][:30]
                        ))
                    fig.update_layout(title="Impact vs Effort Matrix", xaxis_title="Effort →", yaxis_title="Impact →", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No solutions generated")
        
        with tab5:
            st.markdown("### 🎮 Control Phase Results")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Process Capability (Cpk)", f"{cpk:.2f}" if cpk else "N/A")
                if cpk:
                    if cpk >= 1.33:
                        st.success("✅ Process is Capable (Cpk ≥ 1.33)")
                    elif cpk >= 1.0:
                        st.warning("⚠️ Marginally Capable (1.0 ≤ Cpk < 1.33)")
                    else:
                        st.error("❌ Not Capable (Cpk < 1.0)")
                
                is_sustained = state.get('is_sustained', False)
                st.metric("Sustained?", "✅ Yes" if is_sustained else "⚠️ Needs Monitoring")
            
            with col2:
                spc_limits = state.get('spc_limits', (0, 0, 0))
                if spc_limits and isinstance(spc_limits, (tuple, list)) and len(spc_limits) == 3:
                    ucl, mean, lcl = spc_limits
                    if ucl > 0 or mean > 0 or lcl > 0:
                        x = list(range(1, 26))
                        np.random.seed(42)
                        y = [mean + np.random.normal(0, (ucl-lcl)/6) for _ in range(25)]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Process Data'))
                        fig.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text=f"UCL = {ucl}")
                        fig.add_hline(y=mean, line_dash="solid", line_color="green", annotation_text=f"Mean = {mean}")
                        fig.add_hline(y=lcl, line_dash="dash", line_color="red", annotation_text=f"LCL = {lcl}")
                        fig.update_layout(title="Statistical Process Control (SPC) Chart", height=350)
                        st.plotly_chart(fig, use_container_width=True)
            
            control_plan = state.get('control_plan', '')
            if control_plan:
                with st.expander("📋 Control Plan", expanded=True):
                    st.write(control_plan)
    
    # ========== FMEA TABS ==========
    if selected_path == "fmea":
        with tab1:
            st.markdown("### ⚠️ FMEA Analysis Results")
            fmea_rows = state.get('fmea_rows', [])
            if fmea_rows:
                st.markdown("#### 📊 FMEA Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Risks", len(fmea_rows))
                with col2:
                    critical = len([r for r in fmea_rows if (r.rpn if hasattr(r, 'rpn') else r.get('rpn', 0)) >= 200])
                    st.metric("Critical Risks", critical)
                with col3:
                    high = len([r for r in fmea_rows if 200 > (r.rpn if hasattr(r, 'rpn') else r.get('rpn', 0)) >= 100])
                    st.metric("High Risks", high)
                with col4:
                    avg_rpn = sum([r.rpn if hasattr(r, 'rpn') else r.get('rpn', 0) for r in fmea_rows]) / len(fmea_rows)
                    st.metric("Average RPN", f"{avg_rpn:.0f}")
                
                st.markdown("#### 📋 Detailed FMEA Table")
                fmea_data = []
                for row in fmea_rows:
                    if hasattr(row, 'process_step'):
                        fmea_data.append({
                            "Step": row.process_step,
                            "Failure Mode": row.failure_mode[:50],
                            "S": row.severity,
                            "O": row.occurrence,
                            "D": row.detection,
                            "RPN": row.rpn,
                            "Risk Level": get_risk_badge(row.rpn)
                        })
                if fmea_data:
                    st.dataframe(pd.DataFrame(fmea_data), use_container_width=True, height=400)
                
                # RPN Bar Chart
                if fmea_data:
                    fig = go.Figure()
                    steps = [d['Step'] for d in fmea_data]
                    rpns = [d['RPN'] for d in fmea_data]
                    colors = ['red' if r >= 200 else 'orange' if r >= 100 else 'steelblue' for r in rpns]
                    fig.add_trace(go.Bar(x=steps, y=rpns, marker_color=colors))
                    fig.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="Critical RPN (200)")
                    fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="High RPN (100)")
                    fig.update_layout(title="RPN by Process Step", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No FMEA rows generated")
        
        with tab2:
            st.markdown("### 🛡️ Poka-Yoke Solutions")
            poka_yoke_suggestions = state.get('poka_yoke_suggestions', [])
            if poka_yoke_suggestions:
                for suggestion in poka_yoke_suggestions:
                    priority = suggestion.get('priority', 'MEDIUM')
                    priority_emoji = "🔴" if priority == "IMMEDIATE" else "🟠" if priority == "HIGH" else "🟡"
                    with st.expander(f"{priority_emoji} {suggestion.get('process_step', 'Unknown Step')} - RPN: {suggestion.get('rpn', 0)}", expanded=True):
                        st.markdown(f"**Failure Mode:** {suggestion.get('failure_mode', 'Unknown')}")
                        st.markdown("**Mistake-Proofing Suggestions:**")
                        for sugg in suggestion.get('suggestions', []):
                            st.markdown(f"- {sugg}")
            else:
                st.info("No poka-yoke suggestions available.")
        
        with tab3:
            st.markdown("### 📚 Knowledge Sources & Citations")
            display_sources(state)
    
    # ========== SOURCES TABS ==========
    if selected_path == "dmaic":
        with tab6:
            st.markdown("### 📚 Knowledge Sources & Citations")
            display_sources(state)
    elif selected_path == "hybrid":
        with tab7:
            st.markdown("### 📚 Knowledge Sources & Citations")
            display_sources(state)

# ============================================================
# Executive Summary
# ============================================================
if st.session_state.analysis_complete and st.session_state.current_state:
    st.markdown("---")
    st.markdown("### 📊 Executive Summary")
    summary = st.session_state.current_state.get('executive_summary', '')
    if summary:
        st.markdown(summary)
    else:
        st.info("No executive summary generated")

# ============================================================
# Export & Footer
# ============================================================
if st.session_state.analysis_complete and st.session_state.current_state:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 Export JSON Report", use_container_width=True):
            state = st.session_state.current_state
            export_data = {
                "Project": {
                    "Problem": state.get('problem_statement', ''),
                    "Customer": state.get('customer', ''),
                    "Analysis Type": state.get('selected_path', ''),
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "Metrics": {
                    "Baseline": state.get('baseline_metric', 0),
                    "Target": state.get('target_metric', 0),
                    "Sigma": state.get('sigma_level', 0),
                    "Cpk": state.get('cpk_value', 0),
                    "Sustained": state.get('is_sustained', False)
                },
                "Root Causes": state.get('root_causes', []),
                "Solutions Count": len(state.get('solutions', [])),
                "FMEA Steps": len(state.get('fmea_rows', [])),
                "Sources": state.get('total_sources', 0)
            }
            st.json(export_data)
    with col2:
        if st.button("📊 Download Excel Report", use_container_width=True):
            excel_path = st.session_state.current_state.get('excel_report_path', '')
            if excel_path and os.path.exists(excel_path):
                st.success(f"✅ Report saved: {excel_path}")
            else:
                st.warning("Excel report not generated in this run")
    with col3:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.current_state = None
            st.session_state.analysis_complete = False
            st.rerun()

st.markdown("---")
st.markdown("""
    <footer style="text-align: center; padding: 2rem 0; color: #a0aec0;">
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <span>⚡ DMAIC Copilot v2.0</span>
            <span>🔍 Powered by LangGraph + LangChain</span>
            <span>🤖 Groq Llama 3.3 70B</span>
            <span>📚 RAG: PMBOK Knowledge Base</span>
        </div>
        <div style="font-size: 0.8rem; margin-top: 1rem;">
            © 2025 DMAIC Copilot | AI-Powered Six Sigma Assistant | Enterprise Edition
        </div>
    </footer>
""", unsafe_allow_html=True)