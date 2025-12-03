# src/utils.py

import streamlit as st
import PyPDF2
import docx
import re
from datetime import datetime
from typing import List, Tuple, Optional
import pandas as pd
import logging

def load_css(file_name: str):
    """Loads a CSS file into the Streamlit app."""
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Continuing without custom styles.")
    except Exception as e:
        st.error(f"Error loading CSS file '{file_name}': {e}")

def extract_text_from_file(uploaded_file) -> str:
    """Extracts text content from various file types."""
    try:
        if uploaded_file is None:
            return ""
        content_type = getattr(uploaded_file, "type", "")
        # plain text
        if content_type == "text/plain" or uploaded_file.name.lower().endswith(".txt"):
            raw = uploaded_file.read()
            # in some streamlit versions uploaded_file.read() returns bytes
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="ignore")
            return str(raw)
        # pdf
        elif content_type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = []
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text.append(page_text)
            return "\n".join(text)
        # docx
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or uploaded_file.name.lower().endswith(".docx"):
            doc = docx.Document(uploaded_file)
            return "\n".join(para.text for para in doc.paragraphs)
        else:
            # Try to guess by name extension
            name = getattr(uploaded_file, "name", "")
            if name.lower().endswith(".txt"):
                raw = uploaded_file.read()
                if isinstance(raw, bytes):
                    return raw.decode("utf-8", errors="ignore")
                return str(raw)
            st.warning(f"Unsupported file type: {content_type} / {name}")
            return ""
    except Exception as e:
        st.error(f"Error reading file '{getattr(uploaded_file, 'name', 'unknown')}': {e}")
    return ""

def parse_filenames_for_dates(files: List) -> Tuple[List[str], List[datetime]]:
    """Parses filenames to extract dates based on YYYY-MM-DD format.

    Returns:
        texts: list of extracted document texts (in same order as timestamps)
        timestamps: list of datetime objects corresponding to each text
    """
    texts = []
    timestamps = []
    failed_files = []

    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')

    for file in files:
        match = date_pattern.search(getattr(file, "name", ""))
        if match:
            try:
                date_str = match.group(1)
                parsed = datetime.strptime(date_str, "%Y-%m-%d")
                text = extract_text_from_file(file)
                if text and text.strip():
                    timestamps.append(parsed)
                    texts.append(text)
                else:
                    failed_files.append(file.name)
            except ValueError:
                failed_files.append(file.name)
        else:
            failed_files.append(file.name)

    if failed_files:
        st.warning(f"Could not find a date in YYYY-MM-DD format or could not read content for: {', '.join(failed_files)}. These files will be skipped.")

    return texts, timestamps

# Add these imports for language detection
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Add these functions to your src/utils.py file:

def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect language of text using langdetect
    Returns (language_code, confidence)
    """
    if not LANGDETECT_AVAILABLE:
        return "en", 1.0  # Default to English if langdetect not available
    
    try:
        # Clean text for better detection
        clean_text = re.sub(r'[^\w\s]', ' ', text[:1000])  # Use first 1000 chars
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if len(clean_text) < 20:
            return "en", 0.5  # Low confidence for short text
        
        # Get language with confidence
        lang_probs = detect_langs(clean_text)
        if lang_probs:
            top_lang = lang_probs[0]
            return top_lang.lang, top_lang.prob
        else:
            return "en", 0.5
            
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en", 0.5  # Default to English on error

def truncate_text_safely(text: str, max_tokens: int = 512) -> str:
    """
    Truncate text to approximate token limit (rough estimation: 1 token ≈ 4 chars)
    """
    try:
        max_chars = max_tokens * 4  # Rough approximation
        if len(text) <= max_chars:
            return text
        
        # Try to cut at sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        
        last_sentence_end = max(last_period, last_exclamation, last_question)
        
        if last_sentence_end > max_chars * 0.7:  # If we found a sentence end in the last 30%
            return truncated[:last_sentence_end + 1]
        else:
            # Cut at word boundary
            last_space = truncated.rfind(' ')
            if last_space > max_chars * 0.8:  # If we found a space in the last 20%
                return truncated[:last_space]
            else:
                return truncated + "..."
                
    except Exception as e:
        logger.warning(f"Error truncating text: {e}")
        return text[:max_tokens * 4]  # Fallback to simple truncation

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for processing long documents
    """
    try:
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [text]
        
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            
            if end >= len(words):
                break
                
            start = end - overlap
            
        return chunks
        
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        return [text]  # Return original text if chunking fails

def parse_filenames_for_dates(filenames: List[str]) -> List[Optional[datetime]]:
    """
    Enhanced function to parse dates from filenames with multiple patterns
    Returns list of datetime objects (None for unparseable filenames)
    """
    dates = []
    
    # Multiple date patterns to try
    patterns = [
        r'(\d{4}-\d{2}-\d{2})',           # YYYY-MM-DD
        r'(\d{2}-\d{2}-\d{4})',           # MM-DD-YYYY
        r'(\d{4}_\d{2}_\d{2})',           # YYYY_MM_DD
        r'(\d{2}_\d{2}_\d{4})',           # MM_DD_YYYY
        r'(\d{4}\d{2}\d{2})',             # YYYYMMDD
        r'(\d{8})',                       # Generic 8-digit date
    ]
    
    date_formats = [
        '%Y-%m-%d',
        '%m-%d-%Y',
        '%Y_%m_%d',
        '%m_%d_%Y',
        '%Y%m%d',
        '%Y%m%d',
    ]
    
    for filename in filenames:
        parsed_date = None
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, filename)
            if match:
                date_str = match.group(1)
                try:
                    parsed_date = datetime.strptime(date_str, date_formats[i])
                    break
                except ValueError:
                    continue
        
        dates.append(parsed_date)
    
    return dates

def validate_text_input(text: str, min_words: int = 10) -> Tuple[bool, str]:
    """
    Validate text input for analysis
    Returns (is_valid, message)
    """
    if not text or not text.strip():
        return False, "Text is empty"
    
    word_count = len(text.split())
    if word_count < min_words:
        return False, f"Text too short. Need at least {min_words} words, got {word_count}"
    
    # Check for mostly non-alphabetic content
    alphabetic_chars = sum(1 for c in text if c.isalpha())
    total_chars = len(text.replace(' ', ''))
    
    if total_chars > 0 and alphabetic_chars / total_chars < 0.5:
        return False, "Text contains mostly non-alphabetic content"
    
    return True, "Valid"

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def clean_text_for_analysis(text: str) -> str:
    """
    Clean and preprocess text for better analysis
    """
    try:
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.!?;:,-]', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([.!?;:,])', r'\1', text)
        text = re.sub(r'([.!?;:,])\s*([.!?;:,])', r'\1 \2', text)
        
        return text.strip()
        
    except Exception as e:
        logger.warning(f"Error cleaning text: {e}")
        return text

def extract_text_metadata(text: str) -> dict:
    """
    Extract metadata from text content
    """
    try:
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Character counts
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        
        # Calculate readability metrics (simplified)
        avg_words_per_sentence = len(words) / max(len(sentences), 1)
        avg_chars_per_word = char_count_no_spaces / max(len(words), 1)
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'character_count': char_count,
            'character_count_no_spaces': char_count_no_spaces,
            'avg_words_per_sentence': round(avg_words_per_sentence, 2),
            'avg_chars_per_word': round(avg_chars_per_word, 2),
            'estimated_reading_time_minutes': max(1, round(len(words) / 200))  # 200 WPM average
        }
        
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        return {
            'word_count': len(text.split()) if text else 0,
            'sentence_count': 0,
            'paragraph_count': 0,
            'character_count': len(text) if text else 0,
            'character_count_no_spaces': 0,
            'avg_words_per_sentence': 0,
            'avg_chars_per_word': 0,
            'estimated_reading_time_minutes': 1
        }