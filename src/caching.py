# src/caching.py

import streamlit as st
from keybert import KeyBERT
from bertopic import BERTopic
import spacy
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

@st.cache_resource
def load_keybert_model(model_name: str) -> KeyBERT:
    """Loads a specific KeyBERT model and caches it."""
    try:
        return KeyBERT(model=model_name)
    except Exception as e:
        logger.error(f"Failed to load KeyBERT model {model_name}: {e}")
        # Fallback to default model
        return KeyBERT()

@st.cache_resource
def load_bertopic_model():
    """Loads and caches the BERTopic model."""
    try:
        return BERTopic(verbose=False)
    except Exception as e:
        logger.error(f"Failed to load BERTopic model: {e}")
        return None

@st.cache_resource
def load_spacy_model(is_multilingual: bool = False):
    """Loads a spaCy model for NER and caches it."""
    try:
        if is_multilingual:
            # Try to load multilingual model first
            model_names = ["xx_core_web_sm", "en_core_web_sm"]
        else:
            model_names = ["en_core_web_sm", "en_core_web_md"]
        
        for model_name in model_names:
            try:
                nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
                return nlp
            except OSError:
                continue
        
        # If no models found, try to use a blank model
        st.warning("⚠️ No spaCy models found. Please install with: python -m spacy download en_core_web_sm")
        return spacy.blank("en")
        
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {e}")
        return spacy.blank("en")

@st.cache_resource
def load_summarizer_pipeline(is_multilingual: bool = False):
    """Loads and caches a Hugging Face summarization pipeline with proper configuration."""
    try:
        if is_multilingual:
            # Use multilingual summarization models
            models = [
                "sshleifer/distilbart-cnn-12-6",  # Lightweight fallback first
                "facebook/mbart-large-50-many-to-many-mmt",
                "csebuetnlp/mT5_multilingual_XLSum"
            ]
        else:
            models = [
                "sshleifer/distilbart-cnn-12-6",
                "facebook/bart-large-cnn",
                "t5-small"
            ]
        
        for model in models:
            try:
                pipe = pipeline(
                    "summarization", 
                    model=model, 
                    device=-1,  # CPU
                    max_length=1024,  # Fix the truncation warning
                    truncation=True
                )
                logger.info(f"Loaded summarizer model: {model}")
                return pipe
            except Exception as e:
                logger.warning(f"Failed to load {model}: {e}")
                continue
        
        st.warning("⚠️ Could not load any summarization models")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load summarizer: {e}")
        return None

@st.cache_resource
def load_sentiment_pipeline(is_multilingual: bool = False):
    """Loads and caches a Hugging Face sentiment analysis pipeline."""
    try:
        if is_multilingual:
            # Use multilingual sentiment models
            models = [
                "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                "nlptown/bert-base-multilingual-uncased-sentiment",
                "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Fallback
            ]
        else:
            models = [
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "distilbert-base-uncased-finetuned-sst-2-english"
            ]
        
        for model in models:
            try:
                pipe = pipeline(
                    "sentiment-analysis", 
                    model=model, 
                    device=-1,  # CPU
                    truncation=True,
                    max_length=512
                )
                logger.info(f"Loaded sentiment model: {model}")
                return pipe
            except Exception as e:
                logger.warning(f"Failed to load {model}: {e}")
                continue
        
        st.warning("⚠️ Could not load any sentiment analysis models")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load sentiment analyzer: {e}")
        return None

@st.cache_resource
def load_multilingual_models():
    """Load multilingual-specific models and cache them."""
    try:
        models = {}
        
        # Load multilingual KeyBERT
        models['keybert_multilingual'] = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
        
        # Load multilingual spaCy model
        models['spacy_multilingual'] = load_spacy_model(is_multilingual=True)
        
        # Load multilingual summarizer
        models['summarizer_multilingual'] = load_summarizer_pipeline(is_multilingual=True)
        
        # Load multilingual sentiment analyzer
        models['sentiment_multilingual'] = load_sentiment_pipeline(is_multilingual=True)
        
        return models
        
    except Exception as e:
        logger.error(f"Failed to load multilingual models: {e}")
        return {}

def get_model_status():
    """Check the status of loaded models."""
    try:
        status = {
            "keybert": False,
            "spacy": False,
            "summarizer": False,
            "sentiment": False,
            "bertopic": False
        }
        
        # Check KeyBERT
        try:
            test_keybert = KeyBERT()
            status["keybert"] = True
        except:
            pass
        
        # Check spaCy
        try:
            test_spacy = spacy.blank("en")
            status["spacy"] = True
        except:
            pass
        
        # Check Transformers
        try:
            test_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            status["sentiment"] = True
            status["summarizer"] = True  # If one works, likely both will
        except:
            pass
        
        # Check BERTopic
        try:
            test_bertopic = BERTopic(verbose=False)
            status["bertopic"] = True
        except:
            pass
        
        loaded_count = sum(status.values())
        total_count = len(status)
        
        return {
            "status": f"{loaded_count}/{total_count} loaded",
            "all_loaded": loaded_count == total_count,
            "details": status
        }
        
    except Exception as e:
        return {
            "status": "Error checking",
            "all_loaded": False,
            "details": {},
            "error": str(e)
        }

# Optional: Add model cleanup function
def clear_model_cache():
    """Clear all cached models to free memory."""
    st.cache_resource.clear()
    st.success("Model cache cleared successfully!")

# Optional: Add model info function
def get_model_info():
    """Get information about available models."""
    return {
        "keybert_models": [
            "all-MiniLM-L6-v2",
            "paraphrase-multilingual-MiniLM-L12-v2", 
            "bert-base-multilingual-cased"
        ],
        "spacy_models": [
            "en_core_web_sm",
            "en_core_web_md", 
            "xx_core_web_sm"
        ],
        "summarizer_models": [
            "sshleifer/distilbart-cnn-12-6",
            "facebook/bart-large-cnn",
            "facebook/mbart-large-50-many-to-many-mmt"
        ],
        "sentiment_models": [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "distilbert-base-uncased-finetuned-sst-2-english"
        ]
    }