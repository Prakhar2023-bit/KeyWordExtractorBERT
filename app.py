import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any
import io
import base64
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Local Imports from src ---
from src.utils import (
    load_css, extract_text_from_file, parse_filenames_for_dates,
    detect_language, truncate_text_safely, chunk_text
)
from src.caching import (
    load_keybert_model, load_bertopic_model, load_spacy_model,
    load_summarizer_pipeline, load_sentiment_pipeline,
    load_multilingual_models, get_model_status
)
from src.processing import (
    KeywordExtractor, CorpusComparer, KnowledgeGraphGenerator,
    Summarizer, SentimentAnalyzer, EntityExtractor
)
from src.visualizations import (
    create_bar_chart, create_wordcloud, render_knowledge_graph,
    create_sentiment_chart, create_sentiment_trend_chart,
    create_entity_chart, create_topic_wordcloud
)
from src.pdf_generator import PDFReportGenerator

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="KeyInsights",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD CSS ---
load_css("style.css")

# --- SESSION STATE INITIALIZATION ---
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "single_doc_results": {},
        "comparison_results": pd.DataFrame(),
        "trend_results": {},
        "current_language": "en",
        "models_loaded": False,
        "processing_status": "Ready",
        "error_log": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- ERROR HANDLER DECORATOR ---
def handle_errors(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.session_state.error_log.append({"time": datetime.now(), "error": error_msg})
            st.error(f"⚠️ An error occurred: {str(e)}")
            return None
    return wrapper

# --- UI HEADER ---
st.markdown(
    "<link href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css' rel='stylesheet'>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
      .centered-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-top: 20px;
      }
      .main-title {
        font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial;
        font-weight: 700;
        font-size: 2.5rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        justify-content: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      .main-title i {
        font-size: 1.5em;
        color: #667eea;
      }
      .subtitle {
        font-size: 1.2rem;
        font-weight: 400;
        margin-top: 0.5rem;
        color: #555;
      }
      .status-badge {
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 5px 0;
      }
      .status-ready {
        background-color: #4CAF50;
        color: white;
      }
      .status-loading {
        background-color: #FF9800;
        color: white;
      }
      .status-error {
        background-color: #F44336;
        color: white;
      }
      .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title + subtitle block
st.markdown( 
    """ 
    <div class="centered-container"> 
    <h1 class="main-title"><i class="fa-brands fa-searchengin fa-sm"></i> KeyInsights</h1>
    <h3 class="subtitle">Extracting keywords, giving insights</h3> </div> """,
    unsafe_allow_html=True,
)
# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<h2><i class="fa-solid fa-sliders"></i> Control Panel</h2>', unsafe_allow_html=True)
    
    # Model Status Badge
    model_status = get_model_status()
    status_class = "status-ready" if model_status["all_loaded"] else "status-loading"
    st.markdown(
        f'<div class="status-badge {status_class}">Models: {model_status["status"]}</div>',
        unsafe_allow_html=True
    )
    
    # Analysis Mode Selection
    app_mode = st.radio(
        "Analysis Mode",
        ["Single Document Analysis", "Comparative Analysis", "Trend Analysis (Corpus)"],
        help="Choose the type of analysis you want to perform."
    )
    
    # Language Settings
    with st.expander("Language & Model Settings", expanded=False):
        # Language detection toggle
        auto_detect_language = st.checkbox("Auto-detect language", value=True, help="Automatically detect document language")
        
        # Model selection based on language requirements
        MODEL_OPTIONS = {
            "English (MiniLM-L6-v2)": "all-MiniLM-L6-v2",
            "Multilingual (XLM-RoBERTa)": "paraphrase-multilingual-MiniLM-L12-v2",
            "Multilingual Heavy (mBERT)": "bert-base-multilingual-cased"
        }
        selected_model_name = st.selectbox("Select Base Model", options=list(MODEL_OPTIONS.keys()), index=1)
        model_name = MODEL_OPTIONS[selected_model_name]
        
        # Manual language override
        if not auto_detect_language:
            manual_language = st.selectbox(
                "Select Language",
                ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ar", "hi"],
                help="Manually specify document language"
            )
            st.session_state.current_language = manual_language
    
    # Extraction Parameters
    with st.expander("Extraction Parameters", expanded=True):
        n_keywords = st.slider("Number of Keywords", 5, 50, 15, help="Number of keywords to extract")
        ngram_range = st.selectbox(
            "N-gram Range", 
            [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)], 
            index=1,
            help="Range of word combinations"
        )
        use_mmr = st.checkbox("Use MMR for Diversity", True, help="Maximize Marginal Relevance for diverse keywords")
        diversity = st.slider("Keyword Diversity", 0.0, 1.0, 0.7, disabled=not use_mmr, help="Higher = more diverse keywords")
        
    # Summary Parameters
    with st.expander("Summary Parameters", expanded=False):
        summary_max_length = st.slider("Summary Max Length (words)", 50, 500, 150)
        summary_min_length = st.slider("Summary Min Length (words)", 20, 200, 50)
        use_extractive = st.checkbox("Use Extractive Summarization", False, help="Extract key sentences vs generate new text")
    
    # Advanced Settings
    with st.expander("Advanced Settings", expanded=False):
        chunk_size = st.number_input("Text Chunk Size (tokens)", 500, 5000, 2000, help="For processing long documents")
        enable_ner = st.checkbox("Enable Named Entity Recognition", True)
        enable_topic_wordclouds = st.checkbox("Generate Topic Word Clouds", True)
        confidence_threshold = st.slider("Language Detection Confidence", 0.5, 1.0, 0.8)
    
    st.markdown("---")
    
    # Clear Session Button
    if st.button("🔄Reset All", use_container_width=True, help="Clear all cached results and start fresh"):
        # Explicitly reset only the result and log keys to their default empty states
        st.session_state.single_doc_results = {}
        st.session_state.comparison_results = pd.DataFrame()
        st.session_state.trend_results = {}
        st.session_state.error_log = []

        # You can also optionally clear the text input widgets by setting them to an empty string
        if "single_text_area" in st.session_state:
            st.session_state.single_text_area = ""
        if "text_a" in st.session_state:
            st.session_state.text_a = ""
        if "text_b" in st.session_state:
            st.session_state.text_b = ""
            
        st.success("✅Session reset successfully!")
        st.rerun()
    
    # Error Log Display
    if st.session_state.error_log:
        with st.expander(f"⚠️ Error Log ({len(st.session_state.error_log)})", expanded=False):
            for error in st.session_state.error_log[-5:]:  # Show last 5 errors
                st.text(f"{error['time'].strftime('%H:%M:%S')} - {error['error']}")
    
    st.markdown("---")
    st.info("💡 Pro tip: Use multilingual models for better cross-language support")

# --- LOAD MODELS WITH CACHING ---
@st.cache_resource(show_spinner=False)
def load_all_models(model_name, is_multilingual=False):
    """Load all required models with proper caching"""
    try:
        with st.spinner("🚀Loading AI models... This may take a moment on first run."):
            models = {}
            progress_bar = st.progress(0)
            
            # Load KeyBERT model
            progress_bar.progress(0.2, "Loading keyword extraction model...")
            models['keybert'] = load_keybert_model(model_name)
            
            # Load spaCy model
            progress_bar.progress(0.4, "Loading NLP pipeline...")
            models['spacy'] = load_spacy_model(is_multilingual)
            
            # Load summarizer
            progress_bar.progress(0.6, "Loading summarization model...")
            models['summarizer'] = load_summarizer_pipeline(is_multilingual)
            
            # Load sentiment analyzer
            progress_bar.progress(0.8, "Loading sentiment analysis model...")
            models['sentiment'] = load_sentiment_pipeline(is_multilingual)
            
            # Load BERTopic for trend analysis
            progress_bar.progress(0.9, "Loading topic modeling...")
            models['bertopic'] = load_bertopic_model()
            
            progress_bar.progress(1.0, "✅ All models loaded successfully!")
            progress_bar.empty()
            
            return models
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        st.error(f"Failed to load models: {e}")
        return None

# Load models based on selection
is_multilingual = "Multilingual" in selected_model_name
models = load_all_models(model_name, is_multilingual)

if models:
    st.session_state.models_loaded = True
    
    # Initialize processors
    extractor = KeywordExtractor()
    comparer = CorpusComparer()
    graph_gen = KnowledgeGraphGenerator()
    summarizer = Summarizer()
    sentiment_analyzer = SentimentAnalyzer()
    entity_extractor = EntityExtractor()
    pdf_generator = PDFReportGenerator()

    # =====================================================================================
    # --- MODE 1: SINGLE DOCUMENT ANALYSIS ---
    # =====================================================================================
    if app_mode == "Single Document Analysis":
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📄 Document Input")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            text_input = st.text_area("Paste your text here...", height=200, key="single_text_area")
        with col2:
            uploaded_file = st.file_uploader(
                "Or upload a file", 
                type=["txt", "pdf", "docx", "md", "csv"], 
                key="single_uploader",
                help="Supports TXT, PDF, DOCX, MD, CSV formats"
            )
        
        analyze_button = st.button("🔍 Analyze Document", use_container_width=True, type="primary")
        
        if analyze_button:
            # Extract text
            text = ""
            if uploaded_file:
                text = handle_errors(extract_text_from_file)(uploaded_file)
                if text is None:
                    st.error("❌ Failed to extract text from file. Please check if the file contains readable text.")
            else:
                text = text_input
            
            # Validate text
            if text and text.strip():
                # Detect language if auto-detect is enabled
                detected_lang = "en"
                lang_confidence = 1.0
                
                if auto_detect_language:
                    detected_lang, lang_confidence = detect_language(text)
                    if lang_confidence < confidence_threshold:
                        st.warning(f"⚠️ Language detection confidence low ({lang_confidence:.2f}). Results may vary.")
                    st.session_state.current_language = detected_lang
                
                # Check text length
                word_count = len(text.split())
                if word_count < 10:
                    st.warning("⚠️ Text is too short for meaningful analysis. Please provide more content.")
                else:
                    with st.spinner("🔬 Running comprehensive analysis..."):
                        progress = st.progress(0)
                        
                        # Process text in chunks if too long
                        if word_count > chunk_size:
                            st.info(f"📊 Processing large document ({word_count} words) in chunks...")
                            text_chunks = chunk_text(text, chunk_size)
                        else:
                            text_chunks = [text]
                        
                        # Keyword extraction
                        progress.progress(0.2, "Extracting keywords...")
                        all_keywords = []
                        for chunk in text_chunks:
                            chunk_keywords = handle_errors(extractor.extract)(
                                models['keybert'], chunk, top_n=n_keywords, 
                                keyphrase_ngram_range=ngram_range,
                                stop_words='english' if detected_lang == 'en' else None, 
                                use_mmr=use_mmr, diversity=diversity
                            )
                            if chunk_keywords:
                                all_keywords.extend(chunk_keywords)
                        
                        # Aggregate and sort keywords
                        keyword_dict = {}
                        for kw, score in all_keywords:
                            keyword_dict[kw] = keyword_dict.get(kw, 0) + score
                        keywords_scores = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)[:n_keywords]
                        
                        # Knowledge graph generation
                        progress.progress(0.4, "Building knowledge graph...")
                        knowledge_graph = handle_errors(graph_gen.build_graph)(models['spacy'], text[:5000])  # Limit for performance
                        
                        # Sentiment analysis
                        progress.progress(0.6, "Analyzing sentiment...")
                        sentiment_scores = handle_errors(sentiment_analyzer.analyze)(models['sentiment'], text[:2000])  # Limit for performance
                        
                        # Document summarization
                        progress.progress(0.8, "Generating summary...")
                        document_summary = None
                        if word_count >= 50:  # Only summarize if text is long enough
                            try:
                                # Clean and prepare text for summarization
                                clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
                                
                                # Use the truncate_text_safely function
                                truncated_text = truncate_text_safely(clean_text, max_tokens=1000)
                                
                                # Ensure we have meaningful content after truncation
                                if len(truncated_text.strip()) > 20:
                                    # Validate and adjust summary parameters
                                    text_word_count = len(truncated_text.split())
                                    adjusted_max_length = min(summary_max_length, text_word_count - 10)
                                    adjusted_min_length = min(summary_min_length, adjusted_max_length - 10)
                                    
                                    # Ensure parameters are reasonable
                                    if adjusted_max_length < 30:
                                        adjusted_max_length = min(100, text_word_count)
                                    if adjusted_min_length < 10:
                                        adjusted_min_length = max(10, adjusted_max_length // 3)
                                    
                                    document_summary = handle_errors(summarizer.summarize)(
                                        models['summarizer'], 
                                        truncated_text,
                                        max_length=adjusted_max_length,
                                        min_length=adjusted_min_length,
                                        extractive=use_extractive
                                    )
                                else:
                                    document_summary = "Text too short after processing for summarization."
                            except Exception as e:
                                logger.error(f"Summarization failed: {e}")
                                # Use extractive fallback
                                try:
                                    sentences = text.split('. ')[:5]  # Take first 5 sentences as fallback
                                    document_summary = '. '.join(sentences) + '.' if sentences else "Unable to generate summary."
                                except:
                                    document_summary = "Unable to generate summary due to processing error."
                        else:
                            document_summary = "Text too short for summarization."

                        # Named Entity Recognition (if enabled)
                        entities = []
                        if enable_ner:
                            progress.progress(0.9, "Extracting entities...")
                            entities = handle_errors(entity_extractor.extract)(models['spacy'], text[:5000])

                        progress.progress(1.0, "Analysis complete!")
                        progress.empty()

                        # Store results
                        st.session_state.single_doc_results = {
                            "keywords": keywords_scores,
                            "graph": knowledge_graph,
                            "sentiment": sentiment_scores,
                            "summary": document_summary,
                            "text": text,
                            "entities": entities,
                            "language": detected_lang,
                            "lang_confidence": lang_confidence,
                            "word_count": word_count
                        }
                                
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display Results
        if st.session_state.single_doc_results:
            st.markdown("<div class='glass-card result-card'>", unsafe_allow_html=True)
            
            # Language info badge
            results = st.session_state.single_doc_results
            lang_info = f"🌐 Language: {results['language'].upper()} (confidence: {results['lang_confidence']:.2%})"
            st.info(lang_info)
            
            # Results header with metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Word Count", f"{results['word_count']:,}")
            with col2:
                st.metric("🔑 Keywords Found", len(results['keywords']))
            with col3:
                if results['entities']:
                    st.metric("🏷️ Entities Detected", len(results['entities']))
            
            # Tabbed results
            tabs = ["📄 Summary", "📊 Keywords", "☁️ Word Cloud", "🕸️ Knowledge Graph", 
                   "🎭 Sentiment", "🏷️ Entities", "📋 Data Table"]
            tab_objects = st.tabs(tabs)
            
            with tab_objects[0]:  # Summary Tab
                with st.expander("📄 Document Summary", expanded=True):
                    summary = results.get("summary", "")
                    if summary and summary != "Text too short for summarization.":
                        st.markdown(
                            f"""
                            <div style="
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                padding: 20px;
                                border-radius: 10px;
                                margin: 10px 0;
                                color: white;
                                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            ">
                                <h4 style="margin-top: 0; color: white;">📝Quick Summary</h4>
                                <p style="line-height: 1.6; margin-bottom: 0; font-size: 16px;">{summary}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Summary metrics
                        if summary != "Text too short for summarization.":
                            summary_word_count = len(summary.split())
                            reading_time = max(1, round(summary_word_count / 200))
                            compression = round((1 - summary_word_count/results['word_count']) * 100, 1)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Summary Words", summary_word_count)
                            with col2:
                                st.metric("Reading Time", f"{reading_time} min")
                            with col3:
                                st.metric("Compression", f"{compression}%")
                    else:
                        st.info("📝 " + summary)
            
            with tab_objects[1]:  # Keywords Tab
                with st.expander("📊 Keyword Analysis", expanded=True):
                    keywords = results["keywords"]
                    if keywords:
                        df = pd.DataFrame(keywords, columns=["Keyword", "Relevance Score"])
                        fig = create_bar_chart(df, "white")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No keywords extracted.")
            
            with tab_objects[2]:  # Word Cloud Tab
                with st.expander("☁️ Keyword Cloud", expanded=True):
                    if results["keywords"]:
                        fig = create_wordcloud(results["keywords"])
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.warning("No keywords for word cloud generation.")
            
            with tab_objects[3]:  # Knowledge Graph Tab
                with st.expander("🕸️ Knowledge Graph", expanded=True):
                    if results.get("graph"):
                        with st.spinner("Rendering interactive graph..."):
                            render_knowledge_graph(results["graph"])
                    else:
                        st.info("Knowledge graph could not be generated.")
            
            with tab_objects[4]:  # Sentiment Tab
                with st.expander("🎭 Sentiment Analysis", expanded=True):
                    sentiment_scores = results.get("sentiment", {})
                    if sentiment_scores:
                        fig = create_sentiment_chart(sentiment_scores, "white")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Sentiment breakdown table
                        st.subheader("Sentiment Breakdown")
                        sentiment_df = pd.DataFrame(list(sentiment_scores.items()), columns=["Sentiment", "Score"])
                        sentiment_df["Percentage"] = (sentiment_df["Score"] * 100).round(2).astype(str) + "%"
                        st.table(sentiment_df[["Sentiment", "Percentage"]])
                    else:
                        st.info("Sentiment analysis unavailable.")
            
            with tab_objects[5]:  # Entities Tab
                if enable_ner:
                    with st.expander("🏷️ Named Entities", expanded=True):
                        entities = results.get("entities", [])
                        if entities:
                            entity_df = pd.DataFrame(entities)
                            fig = create_entity_chart(entity_df)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("Entity Details")
                            st.dataframe(entity_df, use_container_width=True)
                        else:
                            st.info("No named entities detected.")
                else:
                    st.info("Named Entity Recognition is disabled.")
            
            with tab_objects[6]:  # Data Table Tab
                with st.expander("📋 Keywords Data Table", expanded=True):
                    if results["keywords"]:
                        df = pd.DataFrame(results["keywords"], columns=["Keyword", "Relevance Score"])
                        df["Score %"] = (df["Relevance Score"] * 100).round(2)
                        df = df.sort_values("Relevance Score", ascending=False)
                        st.dataframe(df, use_container_width=True, height=400)
                    else:
                        st.warning("No data available.")
            
            # PDF Export Button
            st.markdown("---")
            if st.button("📥 Download Analysis Report (PDF)", use_container_width=True, type="secondary"):
                with st.spinner("Generating PDF report..."):
                    pdf_bytes = pdf_generator.generate_single_doc_report(results)
                    if pdf_bytes:
                        b64 = base64.b64encode(pdf_bytes).decode()
                        href = f'<a href="data:application/pdf;base64,{b64}" download="nlp_analysis_report.pdf">📄 Click to Download PDF Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        st.success("✅ PDF report generated successfully!")
                    else:
                        st.error("Failed to generate PDF report.")
            
            st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================================================
    # --- MODE 2: COMPARATIVE ANALYSIS ---
    # =====================================================================================
    elif app_mode == "Comparative Analysis":
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("📄 Document A")
            text_a = st.text_area("Paste text A", height=200, key="text_a")
            file_a = st.file_uploader("Or upload file A", type=["txt", "pdf", "docx", "md"], key="file_a")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_b:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("📄 Document B")
            text_b = st.text_area("Paste text B", height=200, key="text_b")
            file_b = st.file_uploader("Or upload file B", type=["txt", "pdf", "docx", "md"], key="file_b")
            st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("🔬 Run Comparative Analysis", use_container_width=True, type="primary"):
            doc_a = handle_errors(extract_text_from_file)(file_a) if file_a else text_a
            doc_b = handle_errors(extract_text_from_file)(file_b) if file_b else text_b
            
            if doc_a and doc_a.strip() and doc_b and doc_b.strip():
                # Validate document lengths
                if len(doc_a.split()) < 10 or len(doc_b.split()) < 10:
                    st.warning("⚠️ Documents are too short for meaningful comparison.")
                else:
                    with st.spinner("Running statistical comparison..."):
                        progress = st.progress(0)
                        
                        # Detect languages
                        progress.progress(0.2, "Detecting languages...")
                        lang_a, conf_a = detect_language(doc_a)
                        lang_b, conf_b = detect_language(doc_b)
                        
                        if lang_a != lang_b:
                            st.warning(f"⚠️ Documents appear to be in different languages ({lang_a} vs {lang_b})")
                        
                        # Calculate keyness
                        progress.progress(0.5, "Calculating keyness scores...")
                        comparison_df = handle_errors(comparer.calculate_keyness)(doc_a, doc_b)
                        
                        # Extract keywords from both
                        progress.progress(0.7, "Extracting keywords...")
                        keywords_a = handle_errors(extractor.extract)(
                            models['keybert'], doc_a, top_n=n_keywords, 
                            keyphrase_ngram_range=ngram_range,
                            use_mmr=use_mmr, diversity=diversity
                        )
                        keywords_b = handle_errors(extractor.extract)(
                            models['keybert'], doc_b, top_n=n_keywords,
                            keyphrase_ngram_range=ngram_range,
                            use_mmr=use_mmr, diversity=diversity
                        )
                        
                        # Sentiment comparison
                        progress.progress(0.9, "Comparing sentiments...")
                        sentiment_a = handle_errors(sentiment_analyzer.analyze)(models['sentiment'], doc_a[:2000])
                        sentiment_b = handle_errors(sentiment_analyzer.analyze)(models['sentiment'], doc_b[:2000])
                        
                        progress.progress(1.0, "✅ Comparison complete!")
                        progress.empty()
                        
                        st.session_state.comparison_results = {
                            "keyness": comparison_df,
                            "keywords_a": keywords_a,
                            "keywords_b": keywords_b,
                            "sentiment_a": sentiment_a,
                            "sentiment_b": sentiment_b,
                            "lang_a": lang_a,
                            "lang_b": lang_b,
                            "doc_a": doc_a,
                            "doc_b": doc_b
                        }
            else:
                st.warning("⚠️ Please provide text for both documents.")
                st.session_state.comparison_results = {}
        
        # Display comparison results
        if (hasattr(st.session_state, 'comparison_results') and 
            st.session_state.comparison_results is not None and 
            isinstance(st.session_state.comparison_results, dict) and 
            bool(st.session_state.comparison_results)):
            st.markdown("<div class='glass-card result-card'>", unsafe_allow_html=True)
            st.subheader("📊 Comparative Analysis Results")
            results = st.session_state.comparison_results
            
            # Language info
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📄 Document A Language: {results['lang_a'].upper()}")
            with col2:
                st.info(f"📄 Document B Language: {results['lang_b'].upper()}")
            # Comparison tabs
            comp_tabs = ["📈 Keyness Analysis", "🔑 Keywords Comparison", "🎭 Sentiment Comparison", "📊 Statistics"]
            comp_tab_objects = st.tabs(comp_tabs)
            
            with comp_tab_objects[0]:  # Keyness Analysis
                with st.expander("📈 Statistical Keyness (Log-Likelihood)", expanded=True):
                    keyness_df = results.get("keyness")
                    if keyness_df is not None and not keyness_df.empty:
                        st.dataframe(keyness_df.head(20), use_container_width=True)
                        
                        # Key findings
                        if len(keyness_df) > 0:
                            st.subheader("📌Key Findings")
                            col1, col2 = st.columns(2)
                            
                        with col1:
                            st.markdown("**🔴 More prominent in Document A:**")
                            # Use the correct column name from the DataFrame
                            keyness_col = "Keyness (Log-Likelihood)"
                            doc_a_keys = keyness_df[keyness_df[keyness_col] > 0].head(5)
                            for _, row in doc_a_keys.iterrows():
                                st.write(f"• {row['Keyword']} (score: {row[keyness_col]:.2f})")

                        with col2:
                            st.markdown("**🔵 More prominent in Document B:**")
                            keyness_col = "Keyness (Log-Likelihood)"
                            doc_b_keys = keyness_df[keyness_df[keyness_col] < 0].head(5)
                            for _, row in doc_b_keys.iterrows():
                                st.write(f"• {row['Keyword']} (score: {row[keyness_col]:.2f})")
                    else:
                        st.info("Keyness analysis could not be completed.")
                        
            with comp_tab_objects[1]:  # Keywords Comparison
                with st.expander("🔑 Top Keywords Side-by-Side", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📄 Document A Keywords**")
                        if results.get("keywords_a"):
                            # Fix: Use "Relevance" instead of "Score" to match create_bar_chart expectations
                            df_a = pd.DataFrame(results["keywords_a"], columns=["Keyword", "Relevance"])
                            fig_a = create_bar_chart(df_a, "lightcoral")
                            st.plotly_chart(fig_a, use_container_width=True)
                        else:
                            st.info("No keywords extracted from Document A")
                    
                    with col2:
                        st.markdown("**📄 Document B Keywords**")
                        if results.get("keywords_b"):
                            # Fix: Use "Relevance" instead of "Score" to match create_bar_chart expectations
                            df_b = pd.DataFrame(results["keywords_b"], columns=["Keyword", "Relevance"])
                            fig_b = create_bar_chart(df_b, "lightblue")
                            st.plotly_chart(fig_b, use_container_width=True)
                        else:
                            st.info("No keywords extracted from Document B")

            with comp_tab_objects[2]:  # Sentiment Comparison
                with st.expander("🎭 Sentiment Comparison", expanded=True):
                    sentiment_a = results.get("sentiment_a", {})
                    sentiment_b = results.get("sentiment_b", {})
                    
                    if sentiment_a and sentiment_b:
                        # Create comparison dataframe
                        sentiment_comparison = pd.DataFrame({
                            'Document A': sentiment_a,
                            'Document B': sentiment_b
                        }).fillna(0)
                        
                        # Side-by-side sentiment charts
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_a = create_sentiment_chart(sentiment_a, "lightcoral")
                            st.plotly_chart(fig_a, use_container_width=True)
                        
                        with col2:
                            fig_b = create_sentiment_chart(sentiment_b, "lightblue")
                            st.plotly_chart(fig_b, use_container_width=True)
                        
                        # Sentiment difference table
                        st.subheader("📊 Sentiment Score Comparison")
                        sentiment_comparison['Difference (A-B)'] = sentiment_comparison['Document A'] - sentiment_comparison['Document B']
                        st.dataframe(sentiment_comparison.round(3), use_container_width=True)
                    else:
                        st.info("Sentiment analysis could not be completed for both documents.")
            
            with comp_tab_objects[3]:  # Statistics
                with st.expander("📊 Document Statistics", expanded=True):
                    doc_a = results.get("doc_a", "")
                    doc_b = results.get("doc_b", "")
                    
                    # Calculate statistics
                    stats_a = {
                        "Word Count": len(doc_a.split()),
                        "Character Count": len(doc_a),
                        "Sentences": len([s for s in doc_a.split('.') if s.strip()]),
                        "Avg Words/Sentence": round(len(doc_a.split()) / max(1, len([s for s in doc_a.split('.') if s.strip()])), 2)
                    }
                    
                    stats_b = {
                        "Word Count": len(doc_b.split()),
                        "Character Count": len(doc_b),
                        "Sentences": len([s for s in doc_b.split('.') if s.strip()]),
                        "Avg Words/Sentence": round(len(doc_b.split()) / max(1, len([s for s in doc_b.split('.') if s.strip()])), 2)
                    }
                    
                    stats_df = pd.DataFrame({
                        'Document A': stats_a,
                        'Document B': stats_b
                    })
                    stats_df['Difference'] = stats_df['Document A'] - stats_df['Document B']
                    
                    st.dataframe(stats_df, use_container_width=True)
            
            # PDF Export for comparison
            st.markdown("---")
            if st.button("📥 Download Comparison Report (PDF)", use_container_width=True, type="secondary"):
                with st.spinner("Generating comparison PDF..."):
                    pdf_bytes = pdf_generator.generate_comparison_report(results)
                    if pdf_bytes:
                        b64 = base64.b64encode(pdf_bytes).decode()
                        href = f'<a href="data:application/pdf;base64,{b64}" download="comparison_analysis_report.pdf">📄 Click to Download Comparison PDF</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        st.success("✅ Comparison PDF generated successfully!")
                    else:
                        st.error("Failed to generate comparison PDF.")
            
            st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================================================
    # --- MODE 3: TREND ANALYSIS (CORPUS) ---
    # =====================================================================================
    elif app_mode == "Trend Analysis (Corpus)":
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📚 Corpus Upload & Analysis")
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload multiple documents for trend analysis",
            type=["txt", "pdf", "docx", "md"],
            accept_multiple_files=True,
            key="corpus_uploader",
            help="Upload 3+ documents to analyze trends over time or across documents"
        )
        
        # Date extraction method
        date_method = st.radio(
            "📅 Date Extraction Method",
            ["Auto-detect from filenames", "Use upload order", "Manual date entry"],
            help="How to determine document chronology"
        )
        
        analyze_corpus_button = st.button("🔍 Analyze Corpus Trends", use_container_width=True, type="primary")
        
        if analyze_corpus_button and uploaded_files:
            if len(uploaded_files) < 2:
                st.warning("⚠️ Please upload at least 2 documents for trend analysis.")
            else:
                with st.spinner("Processing corpus and analyzing trends..."):
                    progress = st.progress(0)
                    
                    # Extract texts and dates
                    progress.progress(0.1, "Extracting text from documents...")
                    corpus_data = []
                    
                    for i, file in enumerate(uploaded_files):
                        text = handle_errors(extract_text_from_file)(file)
                        if text:
                            # Extract date based on method
                            if date_method == "Auto-detect from filenames":
                                date_info = parse_filenames_for_dates([file.name])
                                doc_date = date_info[0] if date_info else datetime.now()
                            elif date_method == "Use upload order":
                                doc_date = datetime.now().replace(day=1) + pd.DateOffset(months=i)
                            else:  # Manual - for now use upload order as fallback
                                doc_date = datetime.now().replace(day=1) + pd.DateOffset(months=i)
                            
                            corpus_data.append({
                                'filename': file.name,
                                'text': text,
                                'date': doc_date,
                                'word_count': len(text.split())
                            })
                    
                    if not corpus_data:
                        st.error("❌ No valid documents could be processed.")
                    else:
                        # Sort by date
                        corpus_data.sort(key=lambda x: x['date'])
                        
                        # Extract keywords for each document
                        progress.progress(0.3, "Extracting keywords from each document...")
                        all_keywords_by_doc = []
                        
                        for doc in corpus_data:
                            keywords = handle_errors(extractor.extract)(
                                models['keybert'], doc['text'], 
                                top_n=n_keywords,
                                keyphrase_ngram_range=ngram_range,
                                use_mmr=use_mmr, diversity=diversity
                            )
                            all_keywords_by_doc.append({
                                'filename': doc['filename'],
                                'date': doc['date'],
                                'keywords': keywords or [],
                                'word_count': doc['word_count']
                            })
                        
                        # Topic modeling on the corpus
                        progress.progress(0.5, "Performing topic modeling...")
                        corpus_texts = [doc['text'] for doc in corpus_data]
                        
                        try:
                            # Fit BERTopic model
                            topics, probs = models['bertopic'].fit_transform(corpus_texts)
                            topic_info = models['bertopic'].get_topic_info()
                            
                            # Get topics for each document
                            doc_topics = []
                            for i, doc in enumerate(corpus_data):
                                doc_topic = topics[i] if i < len(topics) else -1
                                topic_words = models['bertopic'].get_topic(doc_topic) if doc_topic != -1 else []
                                doc_topics.append({
                                    'filename': doc['filename'],
                                    'date': doc['date'],
                                    'topic_id': doc_topic,
                                    'topic_words': topic_words[:5],  # Top 5 words
                                    'probability': probs[i] if i < len(probs) else 0
                                })
                        except Exception as e:
                            logger.warning(f"Topic modeling failed: {e}")
                            doc_topics = []
                            topic_info = pd.DataFrame()
                        
                        # Sentiment trend analysis
                        progress.progress(0.7, "Analyzing sentiment trends...")
                        sentiment_trends = []
                        
                        for doc in corpus_data:
                            sentiment = handle_errors(sentiment_analyzer.analyze)(
                                models['sentiment'], doc['text'][:2000]
                            )
                            sentiment_trends.append({
                                'filename': doc['filename'],
                                'date': doc['date'],
                                'sentiment': sentiment or {}
                            })
                        
                        # Keyword trend analysis
                        progress.progress(0.9, "Calculating keyword trends...")
                        
                        # Track keyword evolution
                        all_keywords = set()
                        for doc_kw in all_keywords_by_doc:
                            all_keywords.update([kw[0] for kw in doc_kw['keywords']])
                        
                        keyword_trends = []
                        for keyword in list(all_keywords)[:50]:  # Limit for performance
                            trend_data = []
                            for doc_kw in all_keywords_by_doc:
                                score = 0
                                for kw, sc in doc_kw['keywords']:
                                    if kw == keyword:
                                        score = sc
                                        break
                                trend_data.append({
                                    'date': doc_kw['date'],
                                    'filename': doc_kw['filename'],
                                    'score': score
                                })
                            
                            keyword_trends.append({
                                'keyword': keyword,
                                'trend_data': trend_data
                            })
                        
                        progress.progress(1.0, "✅ Trend analysis complete!")
                        progress.empty()
                        
                        # Store results
                        st.session_state.trend_results = {
                            'corpus_data': corpus_data,
                            'keywords_by_doc': all_keywords_by_doc,
                            'sentiment_trends': sentiment_trends,
                            'keyword_trends': keyword_trends,
                            'topics': doc_topics,
                            'topic_info': topic_info
                        }
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display trend results
        if st.session_state.trend_results:
            st.markdown("<div class='glass-card result-card'>", unsafe_allow_html=True)
            st.subheader("📈 Corpus Trend Analysis Results")
            
            results = st.session_state.trend_results
            
            # Corpus overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📚 Documents", len(results['corpus_data']))
            with col2:
                total_words = sum(doc['word_count'] for doc in results['corpus_data'])
                st.metric("📝 Total Words", f"{total_words:,}")
            with col3:
                unique_keywords = set()
                for doc in results['keywords_by_doc']:
                    unique_keywords.update([kw[0] for kw in doc['keywords']])
                st.metric("🔑 Unique Keywords", len(unique_keywords))
            with col4:
                if results['topic_info'] is not None and not results['topic_info'].empty:
                    st.metric("🏷️ Topics Found", len(results['topic_info']) - 1)  # Excluding outliers
                else:
                    st.metric("🏷️ Topics Found", "N/A")
            
            # Trend analysis tabs
            trend_tabs = ["📊 Overview", "📈 Sentiment Trends", "🔑 Keyword Evolution", "🏷️ Topic Analysis", "📋 Document Details"]
            trend_tab_objects = st.tabs(trend_tabs)
            
            with trend_tab_objects[0]:  # Overview
                with st.expander("📊 Corpus Overview", expanded=True):
                    # Timeline visualization
                    timeline_data = []
                    for doc in results['corpus_data']:
                        timeline_data.append({
                            'Date': doc['date'],
                            'Document': doc['filename'],
                            'Word Count': doc['word_count']
                        })
                    
                    timeline_df = pd.DataFrame(timeline_data)
                    
                    # Word count over time
                    import plotly.express as px
                    fig = px.line(timeline_df, x='Date', y='Word Count', 
                                 title='Document Word Count Over Time',
                                 hover_data=['Document'])
                    fig.update_layout(template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
            
            with trend_tab_objects[1]:  # Sentiment Trends
                with st.expander("📈 Sentiment Evolution", expanded=True):
                    sentiment_data = []
                    for item in results['sentiment_trends']:
                        for sentiment, score in item['sentiment'].items():
                            sentiment_data.append({
                                'Date': item['date'],
                                'Document': item['filename'],
                                'Sentiment': sentiment,
                                'Score': score
                            })
                    
                    if sentiment_data:
                        sentiment_df = pd.DataFrame(sentiment_data)
                        fig = create_sentiment_trend_chart(sentiment_df)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Sentiment summary table
                        st.subheader("📊 Sentiment Summary by Document")
                        pivot_df = sentiment_df.pivot_table(
                            index=['Date', 'Document'], 
                            columns='Sentiment', 
                            values='Score', 
                            fill_value=0
                        ).round(3)
                        st.dataframe(pivot_df, use_container_width=True)
                    else:
                        st.info("No sentiment data available for trend analysis.")
            
            with trend_tab_objects[2]:  # Keyword Evolution
                with st.expander("🔑 Keyword Trends", expanded=True):
                    # Select top trending keywords
                    top_keywords = []
                    for kw_trend in results['keyword_trends'][:10]:  # Top 10
                        max_score = max([d['score'] for d in kw_trend['trend_data']], default=0)
                        if max_score > 0:
                            top_keywords.append((kw_trend['keyword'], max_score))
                    
                    if top_keywords:
                        # Keyword selection
                        selected_keywords = st.multiselect(
                            "Select keywords to visualize trends:",
                            [kw[0] for kw in sorted(top_keywords, key=lambda x: x[1], reverse=True)],
                            default=[kw[0] for kw in sorted(top_keywords, key=lambda x: x[1], reverse=True)[:5]]
                        )
                        
                        if selected_keywords:
                            # Create trend visualization
                            trend_data = []
                            for kw_trend in results['keyword_trends']:
                                if kw_trend['keyword'] in selected_keywords:
                                    for point in kw_trend['trend_data']:
                                        trend_data.append({
                                            'Date': point['date'],
                                            'Keyword': kw_trend['keyword'],
                                            'Score': point['score'],
                                            'Document': point['filename']
                                        })
                            
                            if trend_data:
                                trend_df = pd.DataFrame(trend_data)
                                fig = px.line(trend_df, x='Date', y='Score', 
                                             color='Keyword',
                                             title='Keyword Relevance Trends Over Time',
                                             hover_data=['Document'])
                                fig.update_layout(template='plotly_white')
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No significant keyword trends found.")
            
            with trend_tab_objects[3]:  # Topic Analysis
                with st.expander("🏷️ Topic Evolution", expanded=True):
                    if results['topics'] and results['topic_info'] is not None and not results['topic_info'].empty:
                        # Topic distribution
                        topic_counts = {}
                        for doc_topic in results['topics']:
                            topic_id = doc_topic['topic_id']
                            if topic_id != -1:  # Exclude outliers
                                topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
                        
                        if topic_counts:
                            # Topic distribution chart
                            topic_df = pd.DataFrame(list(topic_counts.items()), 
                                                  columns=['Topic ID', 'Document Count'])
                            fig = px.bar(topic_df, x='Topic ID', y='Document Count',
                                        title='Topic Distribution Across Documents')
                            fig.update_layout(template='plotly_white')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Topic details table
                            st.subheader("🔍 Topic Details")
                            topic_details = []
                            for doc_topic in results['topics']:
                                if doc_topic['topic_id'] != -1:
                                    topic_words = ', '.join([word for word, score in doc_topic['topic_words']])
                                    topic_details.append({
                                        'Document': doc_topic['filename'],
                                        'Date': doc_topic['date'],
                                        'Topic ID': doc_topic['topic_id'],
                                        'Top Words': topic_words
                                    })
                            
                            if topic_details:
                                topic_details_df = pd.DataFrame(topic_details)
                                st.dataframe(topic_details_df, use_container_width=True)
                        
                        # Generate topic word clouds if enabled
                        if enable_topic_wordclouds and topic_counts:
                            st.subheader("☁️ Topic Word Clouds")
                            for topic_id in list(topic_counts.keys())[:3]:  # Show top 3 topics
                                topic_words = models['bertopic'].get_topic(topic_id)
                                if topic_words:
                                    fig = create_topic_wordcloud(topic_words, f"Topic {topic_id}")
                                    st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("Topic modeling could not identify clear topics in the corpus.")
            
            with trend_tab_objects[4]:  # Document Details
                with st.expander("📋 Document-by-Document Analysis", expanded=True):
                    # Create comprehensive document table
                    doc_details = []
                    for i, doc in enumerate(results['corpus_data']):
                        doc_keywords = results['keywords_by_doc'][i]['keywords']
                        doc_sentiment = results['sentiment_trends'][i]['sentiment']
                        doc_topic = results['topics'][i] if i < len(results['topics']) else {'topic_id': -1}
                        
                        top_keywords = ', '.join([kw[0] for kw in doc_keywords[:5]])
                        dominant_sentiment = max(doc_sentiment.items(), key=lambda x: x[1])[0] if doc_sentiment else 'N/A'
                        
                        doc_details.append({
                            'Document': doc['filename'],
                            'Date': doc['date'].strftime('%Y-%m-%d'),
                            'Word Count': doc['word_count'],
                            'Top Keywords': top_keywords,
                            'Dominant Sentiment': dominant_sentiment,
                            'Topic ID': doc_topic['topic_id'] if doc_topic['topic_id'] != -1 else 'Outlier'
                        })
                    
                    details_df = pd.DataFrame(doc_details)
                    st.dataframe(details_df, use_container_width=True, height=400)
            
            # Export options
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Download Trend Analysis (PDF)", use_container_width=True, type="secondary"):
                    with st.spinner("Generating trend analysis PDF..."):
                        pdf_bytes = pdf_generator.generate_trend_report(results)
                        if pdf_bytes:
                            b64 = base64.b64encode(pdf_bytes).decode()
                            href = f'<a href="data:application/pdf;base64,{b64}" download="trend_analysis_report.pdf">📄 Click to Download Trend PDF</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            st.success("✅ Trend analysis PDF generated!")
                        else:
                            st.error("Failed to generate PDF.")
            
            with col2:
                if st.button("📊 Export Data (CSV)", use_container_width=True, type="secondary"):
                    # Create comprehensive CSV export
                    export_data = []
                    for i, doc in enumerate(results['corpus_data']):
                        doc_keywords = results['keywords_by_doc'][i]['keywords']
                        doc_sentiment = results['sentiment_trends'][i]['sentiment']
                        
                        export_data.append({
                            'Document': doc['filename'],
                            'Date': doc['date'],
                            'Word_Count': doc['word_count'],
                            'Keywords': '; '.join([f"{kw[0]}({kw[1]:.3f})" for kw in doc_keywords]),
                            **{f'Sentiment_{k}': v for k, v in doc_sentiment.items()}
                        })
                    
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="corpus_analysis_data.csv">📊 Download CSV Data</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("✅ CSV export ready!")
            
            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.error("❌ Failed to load required models. Please check your configuration and try again.")
    st.info("💡 This might be due to missing dependencies or network issues. Please ensure all required packages are installed.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>KeyInsights</strong> -Advanced NLP Analysis Platform</p>
    <p>Keyword Extraction • Summarization • Visualization • Comparative Analysis • Trend Detection</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Performance monitoring (optional)
if st.checkbox("🔧 Show Performance Info", value=False):
    st.subheader("Performance Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Models Loaded", "✅" if st.session_state.models_loaded else "❌")
    with col2:
        st.metric("Memory Usage", "Monitoring Active")
    with col3:
        st.metric("Error Count", len(st.session_state.error_log))
    
    if st.session_state.error_log:
        with st.expander("Recent Errors"):
            for error in st.session_state.error_log[-3:]:
                st.code(f"{error['time']}: {error['error']}")   