"""
CT Learner Pro - Enhanced with Groq AI Feedback & Professional Export
Highlights only problematic phrases, provides precise improvement suggestions.
"""

import os
import io
import re
import json
import tempfile
from typing import List, Dict, Tuple, Any
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# File extraction
import docx
import pdfplumber

# Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ---------------------
# Configuration
# ---------------------
COLOR_SCHEME = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#ffbb78",
    "danger": "#d62728",
    "info": "#17becf",
    "light": "#f8f9fa",
    "dark": "#343a40"
}

PAUL_CT_RUBRIC = {
    "Clarity": {
        "description": "Demonstrate clarity in conversation; provide examples.",
        "feedback_q": "Could you elaborate further; give an example?",
        "patterns": ["for example", "for instance", "e.g.", "such as", "to illustrate"],
        "color": "#FF6B6B",
        "rgb": "rgb(255, 107, 107)"
    },
    "Accuracy": {
        "description": "Provide accurate and verifiable information.",
        "feedback_q": "How could we check or verify that?",
        "patterns": ["http", "www.", "cite", "according to", "%", "data", "study"],
        "color": "#4ECDC4",
        "rgb": "rgb(78, 205, 196)"
    },
    "Relevance": {
        "description": "Respond with related information.",
        "feedback_q": "How does that relate to the problem?",
        "patterns": ["related to", "regarding", "pertaining to", "in relation to"],
        "color": "#45B7D1",
        "rgb": "rgb(69, 183, 209)"
    },
    "Significance": {
        "description": "Identify central ideas; contribute important points.",
        "feedback_q": "Is this the most important problem?",
        "patterns": ["main", "central", "important", "key", "primary", "crucial"],
        "color": "#96CEB4",
        "rgb": "rgb(150, 206, 180)"
    },
    "Logic": {
        "description": "Organize information logically.",
        "feedback_q": "Does this make sense together?",
        "patterns": ["therefore", "because", "thus", "hence", "however", "but"],
        "color": "#FFEAA7",
        "rgb": "rgb(255, 234, 167)"
    },
    "Precision": {
        "description": "Be specific, focused, and avoid redundancy.",
        "feedback_q": "Could you be more specific?",
        "patterns": ["specifically", "exactly", "precisely", "in particular"],
        "color": "#DDA0DD",
        "rgb": "rgb(221, 160, 221)"
    },
    "Fairness": {
        "description": "Be open-minded, consider pros/cons.",
        "feedback_q": "Am I representing others' views sympathetically?",
        "patterns": ["on the other hand", "although", "consider", "pros and cons"],
        "color": "#98D8C8",
        "rgb": "rgb(152, 216, 200)"
    },
    "Depth": {
        "description": "Examine intricacies in the argument.",
        "feedback_q": "What are the complexities?",
        "patterns": ["because", "although", "since", "whereas", "in depth", "complex"],
        "color": "#F7DC6F",
        "rgb": "rgb(247, 220, 111)"
    },
    "Breadth": {
        "description": "Offer or consider alternative views.",
        "feedback_q": "Do we need another perspective?",
        "patterns": ["alternatively", "another view", "different perspective", "in contrast"],
        "color": "#BB8FCE",
        "rgb": "rgb(187, 143, 206)"
    }
}

# ---------------------
# Helper Functions
# ---------------------
def extract_text_from_txt_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except Exception:
        return b.decode("latin-1", errors="ignore")

def extract_text_from_docx_bytes(b: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
        f.write(b)
        f.flush()
        tmp = f.name
    try:
        doc = docx.Document(tmp)
        return "\n".join([p.text for p in doc.paragraphs])
    finally:
        try:
            os.unlink(tmp)
        except:
            pass

def extract_text_from_pdf_bytes(b: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(b)
        f.flush()
        tmp = f.name
    try:
        text_pages = []
        with pdfplumber.open(tmp) as pdf:
            for p in pdf.pages:
                text_pages.append(p.extract_text() or "")
        return "\n".join(text_pages)
    finally:
        try:
            os.unlink(tmp)
        except:
            pass

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[\u200b-\u200d\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sentence_split(text: str) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]

def tokenize_simple(s: str) -> List[str]:
    return re.findall(r"\w+['-]?\w*|\w+", s.lower())

def highlight_problematic_phrases(text: str, selected_standards: List[str]) -> Dict[str, List[Tuple[str, str, str, Tuple[int, int]]]]:
    """
    Returns dict mapping standard -> list of (full_sentence, color, problem_description, (start_char, end_char))
    where the phrase is highlighted.
    """
    highlighted = {standard: [] for standard in selected_standards}
    for standard in selected_standards:
        if standard not in PAUL_CT_RUBRIC:
            continue
        data = PAUL_CT_RUBRIC[standard]
        for pattern in data["patterns"]:
            # Find all occurrences of the pattern in the text (case-insensitive)
            for match in re.finditer(re.escape(pattern), text, re.IGNORECASE):
                start = match.start()
                end = match.end()
                # Find the sentence that contains this match
                sent_start = text.rfind('.', 0, start) + 1
                if sent_start == 0:
                    sent_start = 0
                sent_end = text.find('.', end)
                if sent_end == -1:
                    sent_end = len(text)
                sentence = text[sent_start:sent_end].strip()
                # Avoid duplicates (same phrase in same sentence for same standard)
                # Compare using the first element (the sentence) of each stored tuple
                if not any(existing_sent == sentence for existing_sent, _, _, _ in highlighted[standard]):
                    problem = identify_problem(sentence, standard, pattern)
                    highlighted[standard].append((sentence, data["color"], problem, (start, end)))
    return highlighted

def identify_problem(sentence: str, standard: str, pattern: str) -> str:
    problems = {
        "Clarity": "Lacks clear examples or illustrations",
        "Accuracy": "Missing citations, data, or verifiable sources",
        "Relevance": "Information may not directly address the question",
        "Significance": "Fails to identify central or key ideas",
        "Logic": "Missing logical connectors or reasoning gaps",
        "Precision": "Uses vague or hedging language",
        "Fairness": "Lacks balanced perspective or consideration of alternatives",
        "Depth": "Superficial analysis without examining complexities",
        "Breadth": "Narrow perspective without considering alternatives"
    }
    return problems.get(standard, f"Improve {standard.lower()} in this section")

def heuristic_ct_scores(text: str, selected_standards: List[str]) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, List[Tuple[str, str, str, Tuple[int, int]]]]]:
    sents = sentence_split(text)
    tokens = tokenize_simple(text)
    word_count = len(tokens)
    scores = {}
    suggestions = {}
    highlighted = highlight_problematic_phrases(text, selected_standards)

    if "Clarity" in selected_standards:
        clarity_indicators = ["for example", "for instance", "e.g.", "such as", "to illustrate"]
        scores["Clarity"] = 1.0 if any(phrase in text.lower() for phrase in clarity_indicators) else (0.3 if word_count < 50 else 0.5)
        suggestions["Clarity"] = PAUL_CT_RUBRIC["Clarity"]["feedback_q"]
    if "Accuracy" in selected_standards:
        accuracy_indicators = ["http", "www.", "cite", "according to", "%", "data", "study"]
        scores["Accuracy"] = 1.0 if any(ind in text.lower() for ind in accuracy_indicators) else 0.4
        suggestions["Accuracy"] = PAUL_CT_RUBRIC["Accuracy"]["feedback_q"]
    if "Relevance" in selected_standards:
        if sents:
            first = tokenize_simple(sents[0])
            overlap_counts = sum(1 for sent in sents[1:] if any(w in tokenize_simple(sent) for w in first[:5]))
            scores["Relevance"] = min(1.0, (overlap_counts+1) / max(1, len(sents)))
        else:
            scores["Relevance"] = 0.0
        suggestions["Relevance"] = PAUL_CT_RUBRIC["Relevance"]["feedback_q"]
    if "Significance" in selected_standards:
        sign_ind = ["main", "central", "important", "key", "primary"]
        scores["Significance"] = 1.0 if any(w in text.lower() for w in sign_ind) else min(0.9, 0.6 + 0.01 * (word_count/100))
        suggestions["Significance"] = PAUL_CT_RUBRIC["Significance"]["feedback_q"]
    if "Logic" in selected_standards:
        connectors = ["therefore", "because", "thus", "hence", "however", "but", "consequently"]
        scores["Logic"] = min(1.0, sum(1 for c in connectors if c in text.lower()) * 0.25)
        suggestions["Logic"] = PAUL_CT_RUBRIC["Logic"]["feedback_q"]
    if "Precision" in selected_standards:
        hedges = ["maybe", "perhaps", "might", "could", "seems", "appears"]
        scores["Precision"] = max(0.0, 1.0 - 0.2 * sum(1 for h in hedges if h in text.lower()))
        if word_count < 40:
            scores["Precision"] *= 0.5
        suggestions["Precision"] = PAUL_CT_RUBRIC["Precision"]["feedback_q"]
    if "Fairness" in selected_standards:
        fairness_ind = ["on the other hand", "although", "consider", "pros and cons"]
        scores["Fairness"] = 1.0 if any(p in text.lower() for p in fairness_ind) else 0.45
        suggestions["Fairness"] = PAUL_CT_RUBRIC["Fairness"]["feedback_q"]
    if "Depth" in selected_standards:
        depth_ind = ["because", "although", "since", "whereas", "in depth", "complex"]
        scores["Depth"] = min(1.0, 0.25 * sum(1 for d in depth_ind if d in text.lower()) + 0.3)
        suggestions["Depth"] = PAUL_CT_RUBRIC["Depth"]["feedback_q"]
    if "Breadth" in selected_standards:
        breadth_ind = ["alternatively", "another view", "different perspective", "in contrast"]
        scores["Breadth"] = 1.0 if any(p in text.lower() for p in breadth_ind) else 0.4
        suggestions["Breadth"] = PAUL_CT_RUBRIC["Breadth"]["feedback_q"]

    for k in scores:
        scores[k] = float(max(0.0, min(1.0, scores[k])))
    return scores, suggestions, highlighted

def safe_extract_all_files(files) -> List[Dict[str, Any]]:
    out = []
    for f in files:
        name = getattr(f, "name", "uploaded")
        try:
            b = f.read()
            if name.lower().endswith(".pdf"):
                text = extract_text_from_pdf_bytes(b)
            elif name.lower().endswith(".docx"):
                text = extract_text_from_docx_bytes(b)
            else:
                text = extract_text_from_txt_bytes(b)
            text = clean_text(text)
            if not text:
                st.warning(f"Warning: extracted empty text from {name}.")
            out.append({"filename": name, "text": text})
        except Exception as e:
            st.error(f"Failed to extract {name}: {e}")
            out.append({"filename": name, "text": ""})
    return out

def create_ct_heatmap(ct_scores_list: List[Dict[str, float]], filenames: List[str], selected_standards: List[str]) -> go.Figure:
    standards = [std for std in selected_standards if std in PAUL_CT_RUBRIC]
    scores_matrix = []
    for ct_scores in ct_scores_list:
        row = [ct_scores.get(std, 0) for std in standards]
        scores_matrix.append(row)
    fig = go.Figure(data=go.Heatmap(
        z=scores_matrix,
        x=standards,
        y=filenames,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>'
    ))
    fig.update_layout(title="Critical Thinking Scores Heatmap", height=400)
    return fig

def create_comparison_bar_chart(ct_scores: Dict[str, float], student_name: str, selected_standards: List[str]) -> go.Figure:
    standards = [std for std in selected_standards if std in ct_scores]
    scores = [ct_scores[std] for std in standards]
    colors = [PAUL_CT_RUBRIC[std]["color"] for std in standards]
    fig = go.Figure(data=[go.Bar(x=standards, y=scores, marker_color=colors)])
    fig.update_layout(title=f"Critical Thinking Analysis - {student_name}", yaxis=dict(range=[0, 1]), height=400)
    return fig

# ---------------------
# PDF Export Functions
# ---------------------
def export_to_pdf(results_data: Dict, filename: str):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=24, textColor=colors.HexColor('#1f77b4'), alignment=TA_CENTER)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#2e86ab'))
    subheading_style = ParagraphStyle('CustomSubheading', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#666666'))
    bubble_style = ParagraphStyle('BubbleText', parent=styles['Normal'], fontSize=10, leftIndent=20, rightIndent=20, backColor=colors.HexColor('#f0f2f6'))

    story.append(Paragraph("CT Learner Pro - Critical Thinking Analysis Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subheading_style))
    story.append(Spacer(1, 24))

    # Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Spacer(1, 12))

    avg_ct = np.mean([np.mean(list(s.values())) for s in results_data['ct_scores']])
    total_words = sum(len(s["text"].split()) for s in results_data['submissions'])
    high_ct = sum(1 for scores in results_data['ct_scores'] if np.mean(list(scores.values())) > 0.7)

    summary_data = [
        ["Average CT Score", f"{avg_ct:.2f}"],
        ["Total Words", f"{total_words:,}"],
        ["High CT Scores", f"{high_ct}/{len(results_data['submissions'])}"]
    ]
    summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 24))

    # Per student
    for idx, (meta, ct_scores, ct_suggest, ct_highlights, ai_fb) in enumerate(zip(
        results_data['submissions'],
        results_data['ct_scores'],
        results_data['ct_suggestions'],
        results_data['ct_highlights'],
        results_data.get('ai_feedback', [{}] * len(results_data['submissions']))
    )):
        story.append(PageBreak())
        story.append(Paragraph(f"Student: {meta.get('filename', 'Untitled')}", heading_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Score Breakdown", subheading_style))
        story.append(Spacer(1, 6))

        score_data = [["Standard", "Score", "Status"]]
        for std, score in ct_scores.items():
            if std in results_data['selected_standards']:
                status = "✅ Good" if score >= 0.7 else "⚠️ Needs Improvement" if score >= 0.6 else "❌ Critical"
                score_data.append([std, f"{score:.2f}", status])

        score_table = Table(score_data, colWidths=[1.5*inch, 1*inch, 1.5*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 12))

        story.append(Paragraph("Problem Areas & Improvement Suggestions", subheading_style))
        story.append(Spacer(1, 6))

        problem_count = 0
        for std, highlights in ct_highlights.items():
            if highlights and ct_scores.get(std, 0) < 0.7:
                problem_count += 1
                for sent, color, problem, _ in highlights[:2]:
                    bubble_text = f"<b>{std}</b>: {problem}<br/><br/><b>Original Text:</b> \"{sent[:200]}\"<br/><br/><b>Suggestion:</b> {ct_suggest.get(std, 'Review this standard')}"
                    story.append(Paragraph(bubble_text, bubble_style))
                    story.append(Spacer(1, 6))

        if problem_count == 0:
            story.append(Paragraph("No major problem areas identified. Great work!", bubble_style))

        if ai_fb:
            story.append(Paragraph("AI-Generated Feedback", subheading_style))
            story.append(Spacer(1, 6))
            for std, feedback in ai_fb.items():
                if std in results_data['selected_standards']:
                    story.append(Paragraph(f"<b>{std}:</b> {feedback}", bubble_style))
                    story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ---------------------
# Groq AI Feedback Integration
# ---------------------
def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets["GROQ_API_KEY"]
        except:
            pass
    return key

def get_ai_feedback(text: str, selected_standards: List[str]) -> Dict[str, str]:
    if not GROQ_AVAILABLE:
        return {std: "Groq library not installed." for std in selected_standards}
    api_key = get_groq_api_key()
    if not api_key:
        return {std: "Groq API key not configured." for std in selected_standards}
    client = Groq(api_key=api_key)
    standards_list = "\n".join([f"- {std}" for std in selected_standards])
    prompt = f"""You are a critical thinking educator. Analyze the student's text according to the Paul-Elder framework, focusing on these standards:
{standards_list}

Provide concise, actionable feedback for each standard. Keep each feedback to 2-3 sentences.

Student text:
{text[:4000]}

Output format:
[STANDARD]: feedback
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # Updated to currently active model llama-3.3-70b-versatile llama-4-scout-17b-16e-instruct 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        output = response.choices[0].message.content
        feedback_dict = {}
        for line in output.split("\n"):
            if ":" in line:
                std, fb = line.split(":", 1)
                std = std.strip().strip("[]")
                if std in selected_standards:
                    feedback_dict[std] = fb.strip()
        for std in selected_standards:
            if std not in feedback_dict:
                feedback_dict[std] = "No specific feedback generated."
        return feedback_dict
    except Exception as e:
        st.error(f"AI feedback error: {e}")
        return {std: f"Error: {str(e)}" for std in selected_standards}

# ---------------------
# State Management
# ---------------------
def init_session_state():
    defaults = {
        'analysis_results': None,
        'uploaded_files': None,
        'selected_standards': list(PAUL_CT_RUBRIC.keys()),
        'analysis_history': [],
        'current_batch_id': None,
        'processing_status': 'idle',
        'ai_feedback_enabled': False,
        'groq_api_key_input': '',
        'analysis_data': None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

# ---------------------
# Main UI
# ---------------------
def main():
    st.set_page_config(page_title="CT Learner Pro", layout="wide", initial_sidebar_state="expanded", page_icon="🧠")
    init_session_state()

    # Custom CSS for styling
    st.markdown(f"""
    <style>
    .main-header {{ font-size: 2.5rem; color: {COLOR_SCHEME['primary']}; text-align: center; margin-bottom: 2rem; }}
    .metric-card {{ background-color: {COLOR_SCHEME['light']}; padding: 1rem; border-radius: 10px; border-left: 4px solid {COLOR_SCHEME['primary']}; margin: 0.5rem 0; }}
    .highlight-phrase {{
        background-color: rgba(255, 255, 0, 0.3);
        font-weight: bold;
        padding: 0 2px;
        border-radius: 3px;
    }}
    .problem-bubble {{
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .improvement-bubble {{
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .standard-color-preview {{ display: inline-block; width: 20px; height: 20px; border-radius: 3px; margin-right: 8px; vertical-align: middle; }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🧠 CT Learner Pro</h1>', unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; margin-bottom: 2rem;'>Advanced Critical Thinking Analysis with AI Feedback</div>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📁 Upload & Settings")
        uploaded = st.file_uploader("Choose student submissions", accept_multiple_files=True, type=['txt','pdf','docx'])
        st.session_state.uploaded_files = uploaded

        st.markdown("---")
        st.subheader("🎯 Analysis Standards")
        selected_standards = st.multiselect(
            "Choose standards:",
            options=list(PAUL_CT_RUBRIC.keys()),
            default=st.session_state.selected_standards,
            format_func=lambda x: f"{x}",
        )
        st.session_state.selected_standards = selected_standards

        with st.expander("🎨 Color Legend"):
            for std in PAUL_CT_RUBRIC:
                color = PAUL_CT_RUBRIC[std]["color"]
                st.markdown(f'<div style="display: flex; align-items: center; margin: 5px 0;"><div style="background-color: {color}; width: 20px; height: 20px; border-radius: 3px; margin-right: 8px;"></div><span>{std}</span></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All"):
                st.session_state.selected_standards = list(PAUL_CT_RUBRIC.keys())
                st.rerun()
        with col2:
            if st.button("Clear All"):
                st.session_state.selected_standards = []
                st.rerun()

        st.markdown("---")
        st.subheader("🤖 AI Feedback")
        ai_enabled = st.checkbox("Enable Groq AI Feedback", value=st.session_state.ai_feedback_enabled)
        st.session_state.ai_feedback_enabled = ai_enabled
        if ai_enabled:
            api_key_input = st.text_input("Groq API Key (optional)", type="password", value=st.session_state.groq_api_key_input)
            if api_key_input:
                st.session_state.groq_api_key_input = api_key_input
                os.environ["GROQ_API_KEY"] = api_key_input
            if not get_groq_api_key():
                st.error("No Groq API key found. Please enter it above or set GROQ_API_KEY environment variable.")

        st.markdown("---")
        run_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True, disabled=len(selected_standards)==0)

    # Main content
    if run_btn:
        if not st.session_state.uploaded_files:
            st.error("❌ Please upload at least one file.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("📂 Processing files...")
        submissions = safe_extract_all_files(st.session_state.uploaded_files)
        progress_bar.progress(30)

        status_text.text("💭 Evaluating critical thinking...")
        ct_scores_all = []
        ct_suggestions_all = []
        ct_highlights_all = []
        ai_feedback_all = []
        for t in [s["text"] for s in submissions]:
            s, sug, highlights = heuristic_ct_scores(t, st.session_state.selected_standards)
            ct_scores_all.append(s)
            ct_suggestions_all.append(sug)
            ct_highlights_all.append(highlights)
            if st.session_state.ai_feedback_enabled:
                ai_fb = get_ai_feedback(t, st.session_state.selected_standards)
            else:
                ai_fb = {}
            ai_feedback_all.append(ai_fb)
        progress_bar.progress(80)

        status_text.text("📊 Compiling results...")
        analysis_data = {
            'submissions': submissions,
            'ct_scores': ct_scores_all,
            'ct_suggestions': ct_suggestions_all,
            'ct_highlights': ct_highlights_all,
            'ai_feedback': ai_feedback_all,
            'timestamp': datetime.now().isoformat(),
            'selected_standards': st.session_state.selected_standards,
        }
        st.session_state.analysis_data = analysis_data
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

    # Display results if available
    if st.session_state.analysis_data is not None:
        data = st.session_state.analysis_data
        submissions = data['submissions']
        ct_scores_all = data['ct_scores']
        ct_suggestions_all = data['ct_suggestions']
        ct_highlights_all = data['ct_highlights']
        ai_feedback_all = data['ai_feedback']
        selected_standards = data['selected_standards']

        st.markdown("---")
        st.header("📈 Analysis Results")
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📝 Submissions", "📤 Export"])

        with tab1:
            st.subheader("Executive Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_ct = np.mean([np.mean(list(s.values())) for s in ct_scores_all])
                st.metric("Average CT Score", f"{avg_ct:.2f}")
            with col2:
                total_words = sum(len(s["text"].split()) for s in submissions)
                st.metric("Total Words", f"{total_words:,}")
            with col3:
                high_ct = sum(1 for scores in ct_scores_all if np.mean(list(scores.values())) > 0.7)
                st.metric("High CT Scores", f"{high_ct}/{len(submissions)}")

            st.markdown("#### 📊 CT Scores Overview")
            fig = create_ct_heatmap(ct_scores_all, [s["filename"] for s in submissions], selected_standards)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 🎯 CT Standards Performance")
            standards_avg = {}
            for std in selected_standards:
                std_scores = [s.get(std, 0) for s in ct_scores_all]
                standards_avg[std] = np.mean(std_scores) if std_scores else 0
            colors = [PAUL_CT_RUBRIC[std]["color"] for std in standards_avg.keys()]
            fig = px.bar(x=list(standards_avg.keys()), y=list(standards_avg.values()), labels={'x':'CT Standard','y':'Average Score'})
            fig.update_layout(yaxis=dict(range=[0,1]))
            fig.update_traces(marker_color=colors)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Detailed Submission Analysis")
            for i, (meta, ct_scores, ct_suggest, ct_highlights, ai_fb) in enumerate(zip(submissions, ct_scores_all, ct_suggestions_all, ct_highlights_all, ai_feedback_all)):
                with st.expander(f"📄 {i+1}. {meta.get('filename','untitled')}", expanded=i==0):
                    # Original document with phrase highlighting
                    st.markdown("#### 📖 Original Document with Problem Highlights")
                    text = meta.get("text", "")
                    if text:
                        # Build a version of the text with spans around problematic phrases
                        highlights_positions = []
                        for std, highlights in ct_highlights.items():
                            for sent, color, problem, (start, end) in highlights:
                                highlights_positions.append((start, end, std, color, problem))
                        # Sort by start position
                        highlights_positions.sort(key=lambda x: x[0])

                        # Build the highlighted text
                        highlighted_text = ""
                        last_idx = 0
                        for start, end, std, color, problem in highlights_positions:
                            # Add text before the highlight
                            if start > last_idx:
                                highlighted_text += text[last_idx:start]
                            # Add highlighted phrase
                            phrase = text[start:end]
                            # Create a span with a translucent background and tooltip
                            highlighted_text += f'<span class="highlight-phrase" style="background-color: {color}40; font-weight: bold;" title="{std}: {problem}">{phrase}</span>'
                            last_idx = end
                        # Add remaining text
                        if last_idx < len(text):
                            highlighted_text += text[last_idx:]

                        st.markdown(highlighted_text, unsafe_allow_html=True)
                    else:
                        st.warning("No text available.")

                    st.markdown("---")

                    # Problem areas with improvement bubbles
                    st.markdown("#### 🔍 Problem Areas & Improvement Suggestions")
                    problem_found = False
                    for std in selected_standards:
                        score = ct_scores.get(std, 0)
                        if score < 0.6:  # Problem area threshold
                            problem_found = True
                            color = PAUL_CT_RUBRIC[std]["color"]

                            st.markdown(f"""
                            <div class="problem-bubble">
                                <strong style="color: {color};">⚠️ {std} - Problem Identified</strong><br>
                                Score: {score:.2f}/1.00 - Needs significant improvement<br>
                                <strong>Issue:</strong> {PAUL_CT_RUBRIC[std]['feedback_q']}
                            </div>
                            """, unsafe_allow_html=True)

                            # Show specific examples from the text
                            if std in ct_highlights and ct_highlights[std]:
                                for sent, color, problem, _ in ct_highlights[std][:2]:
                                    st.markdown(f"""
                                    <div style="margin-left: 20px; padding: 8px; border-left: 2px solid {color};">
                                        <em>📝 Problematic excerpt:</em> "{sent[:150]}..."<br>
                                        <strong>🎯 Specific problem:</strong> {problem}
                                    </div>
                                    """, unsafe_allow_html=True)

                            # Improvement bubble
                            st.markdown(f"""
                            <div class="improvement-bubble">
                                <strong>💡 Suggested Improvement:</strong><br>
                                {ct_suggest.get(std, 'Review this standard carefully.')}
                            </div>
                            """, unsafe_allow_html=True)

                    if not problem_found:
                        st.success("🎉 No major problem areas identified! Great work on all CT standards.")

                    # Score visualization and AI feedback
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 💭 CT Scores")
                        fig = create_comparison_bar_chart(ct_scores, meta.get('filename','Student'), selected_standards)
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        if st.session_state.ai_feedback_enabled and ai_fb:
                            st.markdown("#### 🤖 AI Feedback")
                            for std, fb in ai_fb.items():
                                if std in selected_standards:
                                    color = PAUL_CT_RUBRIC[std]["color"]
                                    st.markdown(f'<div style="border-left: 4px solid {color}; padding-left: 10px; margin:10px 0;"><strong>{std}:</strong> {fb}</div>', unsafe_allow_html=True)

        with tab3:
            st.subheader("📤 Export Results")

            # Build DataFrame for preview
            rows = []
            for meta, ct_scores, ct_suggest, ai_fb in zip(submissions, ct_scores_all, ct_suggestions_all, ai_feedback_all):
                row = {
                    "filename": meta.get("filename","untitled"),
                    "word_count": len(meta.get("text","").split()),
                    "avg_ct_score": np.mean(list(ct_scores.values())) if ct_scores else 0.0,
                    "ct_scores": json.dumps(ct_scores),
                    "ct_suggestions": json.dumps(ct_suggest),
                    "text_preview": meta.get("text","")[:500]
                }
                if st.session_state.ai_feedback_enabled:
                    row["ai_feedback"] = json.dumps(ai_fb)
                rows.append(row)
            df = pd.DataFrame(rows)

            st.markdown("#### Preview of Export Data")
            simple_df = pd.DataFrame({
                "Filename": df["filename"],
                "Word Count": df["word_count"],
                "Overall CT Score": df["avg_ct_score"].round(3),
                "Text Preview": df["text_preview"]
            })
            for std in selected_standards:
                simple_df[std] = df["ct_scores"].apply(lambda x: round(json.loads(x).get(std, 0), 3))
            if st.session_state.ai_feedback_enabled:
                for std in selected_standards:
                    simple_df[f"AI_Feedback_{std}"] = df["ai_feedback"].apply(lambda x: json.loads(x).get(std, "")[:100])
            st.dataframe(simple_df, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Download Options")

            col1, col2 = st.columns(2)

            with col1:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                towrite = io.BytesIO()
                with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Results")
                    ct_details = []
                    for i, (meta, ct_scores, ct_suggest) in enumerate(zip(submissions, ct_scores_all, ct_suggestions_all)):
                        for std in selected_standards:
                            if std in ct_scores:
                                ct_details.append({
                                    "Filename": meta["filename"],
                                    "CT_Standard": std,
                                    "Score": ct_scores[std],
                                    "Suggestion": ct_suggest[std]
                                })
                    pd.DataFrame(ct_details).to_excel(writer, index=False, sheet_name="CT_Details")
                    if st.session_state.ai_feedback_enabled:
                        ai_details = []
                        for meta, ai_fb in zip(submissions, ai_feedback_all):
                            for std, fb in ai_fb.items():
                                if std in selected_standards:
                                    ai_details.append({
                                        "Filename": meta["filename"],
                                        "CT_Standard": std,
                                        "AI_Feedback": fb
                                    })
                        pd.DataFrame(ai_details).to_excel(writer, index=False, sheet_name="AI_Feedback")

                st.download_button(
                    label="📊 Download Excel Report",
                    data=towrite.getvalue(),
                    file_name=f"ctlearner_results_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            with col2:
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    with st.spinner("Generating PDF report..."):
                        pdf_buffer = export_to_pdf(data, f"ct_report_{timestamp}.pdf")
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"ctlearner_report_{timestamp}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

        st.success("🎉 Analysis complete! Explore the results in the tabs above.")

    else:
        st.markdown("---")
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("🚀 Getting Started")
            st.markdown("""
            1. **Select** CT standards to analyze in the sidebar
            2. **Upload** student submissions (TXT, PDF, or DOCX)
            3. **Click** 'Start Analysis' to begin processing
            4. **Explore** color-coded results with problem highlighting
            5. **Export** results as Excel or PDF reports

            ### Features:
            - 🎨 **Color-coded** Paul's CT Standards
            - 🔍 **Phrase-level highlighting** – only problematic parts are marked
            - 💬 **Improvement suggestions** in bubble text boxes
            - 🤖 **AI feedback** with Groq (optional)
            - 📊 **Export** to Excel and PDF formats
            """)
        with col2:
            st.subheader("🎯 CT Standards")
            for std, data in PAUL_CT_RUBRIC.items():
                st.markdown(f'<div style="display: flex; align-items: center; margin:8px 0;"><div style="background-color: {data["color"]}; width:20px; height:20px; border-radius:5px; margin-right:10px;"></div><div><strong>{std}</strong><br><span style="font-size:0.8rem;">{data["description"][:50]}...</span></div></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
