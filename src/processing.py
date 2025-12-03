import streamlit as st
import pandas as pd
import numpy as np
import math
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from itertools import combinations
import spacy


class KeywordExtractor:
    """Handles keyword extraction using KeyBERT."""
    def extract(self, model, text: str, **kwargs) -> List[Tuple[str, float]]:
        try:
            return model.extract_keywords(text, **kwargs)
        except Exception as e:
            st.error(f"An error occurred during keyword extraction: {e}")
            return []


class CorpusComparer:
    """Performs statistical comparison between two corpora using a 2x2 G-test (log-likelihood)."""

    def _g_statistic_2x2(self, a: int, b: int, total_a: int, total_b: int) -> float:
        """
        Compute the log-likelihood G statistic for a 2x2 contingency table:
            [[a, b],
             [total_a - a, total_b - b]]
        """
        N = total_a + total_b
        if N == 0:
            return 0.0

        O11 = float(a)
        O12 = float(b)
        O21 = float(total_a - a)
        O22 = float(total_b - b)

        C1 = O11 + O21
        C2 = O12 + O22
        R1 = O11 + O12
        R2 = O21 + O22

        eps = 1e-12
        E11 = (R1 * C1) / N if N != 0 else eps
        E12 = (R1 * C2) / N if N != 0 else eps
        E21 = (R2 * C1) / N if N != 0 else eps
        E22 = (R2 * C2) / N if N != 0 else eps

        G = 0.0
        for O, E in ((O11, E11), (O12, E12), (O21, E21), (O22, E22)):
            if O > 0 and E > 0:
                G += O * math.log(O / E)

        G *= 2.0
        if not (math.isfinite(G) and G >= 0):
            return 0.0
        return float(G)

    def calculate_keyness(self, doc_a: str, doc_b: str) -> pd.DataFrame:
        """Calculates keyness (log-likelihood G) for each token in the combined vocabulary."""
        try:
            if not isinstance(doc_a, str) or not isinstance(doc_b, str):
                return pd.DataFrame()

            vectorizer = CountVectorizer(stop_words="english", dtype=np.int64)
            docs = [doc_a, doc_b]
            counts = vectorizer.fit_transform(docs)
            if counts.shape[1] == 0:
                return pd.DataFrame()

            vocab = vectorizer.get_feature_names_out()
            arr = counts.toarray()
            freq_a = arr[0].astype(int)
            freq_b = arr[1].astype(int)

            total_a = int(freq_a.sum())
            total_b = int(freq_b.sum())

            if total_a == 0 or total_b == 0:
                return pd.DataFrame()

            results = []
            N = total_a + total_b
            for i, word in enumerate(vocab):
                a = int(freq_a[i])
                b = int(freq_b[i])
                if (a + b) == 0:
                    continue

                g_val = self._g_statistic_2x2(a=a, b=b, total_a=total_a, total_b=total_b)
                expected_a = ((a + b) * total_a) / N if N != 0 else 0.0
                if float(a) < expected_a:
                    g_val = -g_val

                results.append({
                    "Keyword": word,
                    "Frequency A": a,
                    "Frequency B": b,
                    "Keyness (Log-Likelihood)": float(g_val)
                })

            if not results:
                return pd.DataFrame()

            df = pd.DataFrame(results)
            df = df.sort_values(by="Keyness (Log-Likelihood)", ascending=False).reset_index(drop=True)
            return df

        except Exception as e:
            st.error(f"Error during keyness calculation: {e}")
            return pd.DataFrame()


class KnowledgeGraphGenerator:
    """Builds a knowledge graph from text."""
    def build_graph(self, spacy_model, text: str, min_sent_len: int = 5) -> nx.Graph:
        G = nx.Graph()
        try:
            doc = spacy_model(text)
            for sent in doc.sents:
                if len(sent) < min_sent_len:
                    continue

                nodes = set()
                for ent in sent.ents:
                    nodes.add((ent.text.lower().strip(), ent.label_))
                for chunk in sent.noun_chunks:
                    if chunk.root.pos_ != "PRON":
                        nodes.add((chunk.root.text.lower().strip(), "NOUN_CHUNK"))

                if len(nodes) > 1:
                    for node1, node2 in combinations(nodes, 2):
                        term1, type1 = node1
                        term2, type2 = node2
                        if not G.has_node(term1):
                            G.add_node(term1, type=type1)
                        if not G.has_node(term2):
                            G.add_node(term2, type=type2)
                        if G.has_edge(term1, term2):
                            G[term1][term2]["weight"] += 1
                        else:
                            G.add_edge(term1, term2, weight=1)
        except Exception as e:
            st.warning(f"Knowledge graph generation encountered an error: {e}")
        return G


class SentimentAnalyzer:
    """Handles sentiment analysis using a Hugging Face pipeline."""

    def analyze(self, sentiment_pipeline, text: str, **kwargs) -> Dict[str, float]:
        """Analyzes a single text block and returns sentiment probabilities."""
        try:
            if not text:
                return {}
            truncated = text[:512]
            results = sentiment_pipeline(truncated, return_all_scores=True, **kwargs)
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                scores = {item["label"].capitalize(): float(item["score"]) for item in results[0]}
                normalized = {}
                for key in ["Positive", "Neutral", "Negative"]:
                    if key in scores:
                        normalized[key] = scores[key]
                for k, v in scores.items():
                    if k not in normalized:
                        normalized[k] = v
                return normalized
            if isinstance(results, dict):
                return {results.get("label", "Label"): float(results.get("score", 0.0))}
            return {}
        except Exception as e:
            st.error(f"Could not perform sentiment analysis: {e}")
            return {}

    def analyze_corpus(self, sentiment_pipeline, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Analyzes a list of documents and returns the dominant sentiment for each."""
        try:
            if not texts:
                return []
            return sentiment_pipeline(texts, truncation=True, max_length=512, **kwargs)
        except Exception as e:
            st.error(f"Could not perform corpus sentiment analysis: {e}")
            return []

class Summarizer:
    """Generates abstractive summaries for text blocks."""

    def summarize(
        self,
        summarizer_pipeline,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        extractive: bool = False,
        **kwargs
    ) -> str:
        if not text or not text.strip():
            return "No text provided for summarization."
        
        # If no pipeline is available, use extractive fallback
        if summarizer_pipeline is None:
            return self.extractive_summarize(text, num_sentences=3)
        
        try:
            # Clean and validate input text
            clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
            
            # Ensure text is long enough for summarization
            word_count = len(clean_text.split())
            if word_count < 20:
                return "Text too short for summarization."
            
            # If extractive is requested, use simple extractive
            if extractive:
                return self.extractive_summarize(clean_text, num_sentences=5)
            
            # Adjust parameters based on input length
            adjusted_max_length = min(max_length, word_count)
            adjusted_min_length = min(min_length, max(10, adjusted_max_length // 3))
            
            # Ensure min_length < max_length
            if adjusted_min_length >= adjusted_max_length:
                adjusted_min_length = max(10, adjusted_max_length - 20)
            
            # Truncate text if too long (model token limits)
            if word_count > 800:
                words = clean_text.split()
                clean_text = ' '.join(words[:800])
            
            # Call the summarization pipeline with correct parameters
            # Remove any parameters that cause conflicts
            summary_result = summarizer_pipeline(
                clean_text,
                max_length=adjusted_max_length,
                min_length=adjusted_min_length,
                do_sample=False,
                truncation=True
            )
            
            # Extract summary text from result
            if isinstance(summary_result, list) and len(summary_result) > 0:
                summary_text = summary_result[0].get("summary_text", "")
                if summary_text and summary_text.strip():
                    return summary_text.strip()
            elif isinstance(summary_result, dict):
                summary_text = summary_result.get("summary_text", "")
                if summary_text and summary_text.strip():
                    return summary_text.strip()
            
            # Fallback to extractive if AI summarization fails
            return self.extractive_summarize(clean_text, num_sentences=3)
                
        except Exception as e:
            error_msg = f"AI summarization failed, using extractive fallback: {str(e)}"
            st.warning(error_msg)
            # Use extractive fallback
            return self.extractive_summarize(text, num_sentences=3)

    def extractive_summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        Simple extractive summarization by selecting key sentences.
        """
        try:
            if not text or not text.strip():
                return "No text provided for extractive summarization."
            
            # Split into sentences
            import re
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
            
            if len(sentences) <= num_sentences:
                return '. '.join(sentences) + '.'
            
            # Simple scoring based on sentence length and position
            scored_sentences = []
            
            for i, sentence in enumerate(sentences):
                # Length score (prefer medium-length sentences)
                length_score = min(len(sentence.split()) / 25, 1.0)
                if len(sentence.split()) < 5:
                    length_score *= 0.5
                
                # Position score (slight preference for earlier sentences)
                position_score = 1.0 - (i / len(sentences)) * 0.3
                
                total_score = length_score * 0.7 + position_score * 0.3
                scored_sentences.append((sentence, total_score, i))
            
            # Select top sentences and maintain original order
            top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:num_sentences]
            top_sentences = sorted(top_sentences, key=lambda x: x[2])
            
            summary = '. '.join([sent[0] for sent in top_sentences])
            
            if not summary.endswith('.'):
                summary += '.'
                
            return summary
            
        except Exception as e:
            return f"Extractive summarization failed: {str(e)}"

@st.cache_resource
def load_summarizer_pipeline_enhanced(is_multilingual: bool = False):
    """Enhanced summarizer loading with better error handling."""
    import streamlit as st
    from transformers import pipeline
    
    try:
        # List of models to try (in order of preference)
        if is_multilingual:
            model_candidates = [
                "facebook/mbart-large-50-many-to-many-mmt",
                "google/mt5-small",
                "sshleifer/distilbart-cnn-12-6"  # English fallback
            ]
        else:
            model_candidates = [
                "sshleifer/distilbart-cnn-12-6",
                "facebook/bart-large-cnn", 
                "google/pegasus-cnn_dailymail",
                "t5-small"
            ]
        
        for model_name in model_candidates:
            try:
                st.info(f"🤖 Loading summarization model: {model_name}")
                
                # Load with explicit parameters
                summarizer = pipeline(
                    "summarization", 
                    model=model_name,
                    device=-1,  # Use CPU
                    framework="pt",  # PyTorch
                    trust_remote_code=False
                )
                
                # Test the model with a simple example
                test_text = "This is a test sentence. It should work fine for summarization testing purposes."
                test_result = summarizer(test_text, max_length=50, min_length=10, do_sample=False)
                
                if test_result and len(test_result) > 0:
                    st.success(f"✅ Successfully loaded: {model_name}")
                    return summarizer
                    
            except Exception as e:
                st.warning(f"⚠️ Failed to load {model_name}: {str(e)}")
                continue
        
        # If all models failed, return None and handle in the main code
        st.error("❌ Could not load any summarization models")
        return None
        
    except Exception as e:
        st.error(f"❌ Critical error in summarizer loading: {e}")
        return None

# Debug helper function - add this to your main app for testing
def test_summarization_debug():
    """Debug function to test summarization pipeline."""
    st.subheader("🔧 Summarization Debug")
    
    # Test text
    test_text = st.text_area(
        "Test text for summarization:",
        value="This is a longer test text for summarization. It contains multiple sentences that should provide enough content for the summarization model to work with. The goal is to see if the model can create a shorter, meaningful summary of this content. If this works, then the summarization pipeline is functioning correctly.",
        height=100
    )
    
    if st.button("🧪 Test Summarization"):
        try:
            # Load models
            with st.spinner("Loading summarization model..."):
                summarizer_pipeline = load_summarizer_pipeline_enhanced(is_multilingual=False)
            
            if summarizer_pipeline is None:
                st.error("❌ Failed to load summarizer")
                return
            
            # Test summarization
            summarizer = Summarizer()
            
            with st.spinner("Testing summarization..."):
                result = summarizer.summarize(
                    summarizer_pipeline,
                    test_text,
                    max_length=100,
                    min_length=20
                )
            
            st.success("✅ Summarization test completed!")
            st.write("**Result:**", result)
            
        except Exception as e:
            st.error(f"❌ Debug test failed: {e}")
            st.code(str(e))


class EntityExtractor:
    """Handles Named Entity Recognition using spaCy."""
    
    def extract(self, spacy_model, text: str, min_length: int = 2) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using spaCy model.
        
        Args:
            spacy_model: Loaded spaCy model
            text: Input text to analyze
            min_length: Minimum length of entity text to include
            
        Returns:
            List of dictionaries containing entity information
        """
        try:
            if not text or not text.strip():
                return []
            
            # Process text with spaCy
            doc = spacy_model(text)
            entities = []
            
            # Extract entities
            seen_entities = set()  # To avoid duplicates
            
            for ent in doc.ents:
                entity_text = ent.text.strip()
                entity_label = ent.label_
                
                # Filter by minimum length and avoid duplicates
                if len(entity_text) >= min_length and entity_text.lower() not in seen_entities:
                    entities.append({
                        'text': entity_text,
                        'label': entity_label,
                        'description': spacy.explain(entity_label) or entity_label,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': getattr(ent, 'confidence', 1.0)  # Some models provide confidence scores
                    })
                    seen_entities.add(entity_text.lower())
            
            # Sort by start position
            entities.sort(key=lambda x: x['start'])
            
            return entities
            
        except Exception as e:
            st.error(f"Error during entity extraction: {e}")
            return []
    
    def extract_by_type(self, spacy_model, text: str, entity_types: List[str] = None) -> Dict[str, List[str]]:
        """
        Extract entities grouped by type.
        
        Args:
            spacy_model: Loaded spaCy model
            text: Input text to analyze
            entity_types: List of entity types to extract (None for all)
            
        Returns:
            Dictionary mapping entity types to lists of entity texts
        """
        try:
            entities = self.extract(spacy_model, text)
            
            if entity_types:
                # Filter by specified types
                entities = [e for e in entities if e['label'] in entity_types]
            
            # Group by type
            grouped = {}
            for entity in entities:
                entity_type = entity['label']
                if entity_type not in grouped:
                    grouped[entity_type] = []
                grouped[entity_type].append(entity['text'])
            
            return grouped
            
        except Exception as e:
            st.error(f"Error grouping entities by type: {e}")
            return {}
    
    def get_entity_statistics(self, spacy_model, text: str) -> Dict[str, Any]:
        """
        Get statistics about entities in the text.
        
        Returns:
            Dictionary with entity statistics
        """
        try:
            entities = self.extract(spacy_model, text)
            
            if not entities:
                return {
                    'total_entities': 0,
                    'unique_entities': 0,
                    'entity_types': {},
                    'most_common_type': None
                }
            
            # Count by type
            type_counts = {}
            unique_entities = set()
            
            for entity in entities:
                entity_type = entity['label']
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
                unique_entities.add(entity['text'].lower())
            
            # Find most common type
            most_common_type = max(type_counts, key=type_counts.get) if type_counts else None
            
            return {
                'total_entities': len(entities),
                'unique_entities': len(unique_entities),
                'entity_types': type_counts,
                'most_common_type': most_common_type,
                'entities_per_type': {
                    entity_type: [e['text'] for e in entities if e['label'] == entity_type]
                    for entity_type in type_counts.keys()
                }
            }
            
        except Exception as e:
            st.error(f"Error calculating entity statistics: {e}")
            return {
                'total_entities': 0,
                'unique_entities': 0,
                'entity_types': {},
                'most_common_type': None
            }