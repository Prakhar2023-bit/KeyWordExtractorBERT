import io
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image as RLImage
)
import pandas as pd

class PDFReportGenerator:
    """
    Generates comprehensive PDF reports for NLP analysis results.
    """

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the PDF."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            spaceBefore=20,
            textColor=colors.darkblue
        ))
        
        # Subheading style
        self.styles.add(ParagraphStyle(
            name='CustomSubHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            spaceBefore=15,
            textColor=colors.blue
        ))

    def generate_single_doc_report(self, results: Dict[str, Any]) -> Optional[bytes]:
        """Generate PDF report for single document analysis."""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            story = []
            
            # Title
            story.append(Paragraph("Analysis Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 20))
            
            # Document info
            story.append(Paragraph("Document Analysis Summary", self.styles['CustomHeading']))
            
            # Basic stats
            word_count = results.get('word_count', 0)
            language = results.get('language', 'unknown').upper()
            lang_confidence = results.get('lang_confidence', 0) * 100
            
            doc_info = f"""
            <b>Word Count:</b> {word_count:,}<br/>
            <b>Language:</b> {language} (confidence: {lang_confidence:.1f}%)<br/>
            <b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            """
            story.append(Paragraph(doc_info, self.styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Summary section
            summary = results.get('summary', '')
            if summary and summary != "Text too short for summarization.":
                story.append(Paragraph("Document Summary", self.styles['CustomHeading']))
                story.append(Paragraph(summary, self.styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Keywords section
            keywords = results.get('keywords', [])
            if keywords:
                story.append(Paragraph("Top Keywords", self.styles['CustomHeading']))
                
                # Create keywords table
                keyword_data = [['Keyword', 'Relevance Score']]
                for i, (keyword, score) in enumerate(keywords[:15]):  # Top 15
                    keyword_data.append([keyword, f"{score:.3f}"])
                
                keyword_table = Table(keyword_data, colWidths=[3*inch, 1.5*inch])
                keyword_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(keyword_table)
                story.append(Spacer(1, 20))
            
            # Sentiment analysis
            sentiment = results.get('sentiment', {})
            if sentiment:
                story.append(Paragraph("Sentiment Analysis", self.styles['CustomHeading']))
                
                sentiment_data = [['Sentiment', 'Score', 'Percentage']]
                for sent_type, score in sentiment.items():
                    sentiment_data.append([
                        sent_type.capitalize(),
                        f"{score:.3f}",
                        f"{score*100:.1f}%"
                    ])
                
                sentiment_table = Table(sentiment_data, colWidths=[2*inch, 1*inch, 1*inch])
                sentiment_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(sentiment_table)
                story.append(Spacer(1, 20))
            
            # Entities section
            entities = results.get('entities', [])
            if entities:
                story.append(Paragraph("Named Entities", self.styles['CustomHeading']))
                
                # Group entities by type
                entity_types = {}
                for entity in entities:
                    entity_type = entity.get('label', 'UNKNOWN')
                    if entity_type not in entity_types:
                        entity_types[entity_type] = []
                    entity_types[entity_type].append(entity.get('text', ''))
                
                for entity_type, entity_list in entity_types.items():
                    story.append(Paragraph(f"{entity_type}:", self.styles['CustomSubHeading']))
                    entity_text = ", ".join(entity_list[:10])  # Limit to avoid overflow
                    story.append(Paragraph(entity_text, self.styles['Normal']))
                    story.append(Spacer(1, 10))
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"Error generating single document PDF: {e}")
            return None

    def generate_comparison_report(self, results: Dict[str, Any]) -> Optional[bytes]:
        """Generate PDF report for document comparison."""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            story = []
            
            # Title
            story.append(Paragraph("Document Comparison Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 20))
            
            # Document info
            lang_a = results.get('lang_a', 'unknown').upper()
            lang_b = results.get('lang_b', 'unknown').upper()
            
            doc_info = f"""
            <b>Document A Language:</b> {lang_a}<br/>
            <b>Document B Language:</b> {lang_b}<br/>
            <b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            """
            story.append(Paragraph(doc_info, self.styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Keyness analysis
            keyness_df = results.get('keyness')
            if keyness_df is not None and not keyness_df.empty:
                story.append(Paragraph("Statistical Keyness Analysis", self.styles['CustomHeading']))
                story.append(Paragraph(
                    "Positive scores indicate words more prominent in Document A, "
                    "negative scores indicate words more prominent in Document B.",
                    self.styles['Normal']
                ))
                story.append(Spacer(1, 10))
                
                # Top keyness words
                keyness_data = [['Word', 'Freq A', 'Freq B', 'Keyness Score']]
                for _, row in keyness_df.head(20).iterrows():
                    keyness_data.append([
                        row.get('Keyword', ''),
                        str(row.get('Frequency A', 0)),
                        str(row.get('Frequency B', 0)),
                        f"{row.get('Keyness (Log-Likelihood)', 0):.2f}"
                    ])
                
                keyness_table = Table(keyness_data, colWidths=[2*inch, 1*inch, 1*inch, 1.5*inch])
                keyness_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(keyness_table)
                story.append(PageBreak())
            
            # Keywords comparison
            keywords_a = results.get('keywords_a', [])
            keywords_b = results.get('keywords_b', [])
            
            if keywords_a or keywords_b:
                story.append(Paragraph("Keywords Comparison", self.styles['CustomHeading']))
                
                # Side by side keywords
                max_len = max(len(keywords_a), len(keywords_b))
                comparison_data = [['Document A Keywords', 'Score', 'Document B Keywords', 'Score']]
                
                for i in range(min(max_len, 15)):  # Limit to 15 rows
                    kw_a = keywords_a[i] if i < len(keywords_a) else ('', 0)
                    kw_b = keywords_b[i] if i < len(keywords_b) else ('', 0)
                    comparison_data.append([
                        kw_a[0], f"{kw_a[1]:.3f}" if kw_a[1] else '',
                        kw_b[0], f"{kw_b[1]:.3f}" if kw_b[1] else ''
                    ])
                
                comparison_table = Table(comparison_data, colWidths=[2*inch, 1*inch, 2*inch, 1*inch])
                comparison_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(comparison_table)
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"Error generating comparison PDF: {e}")
            return None

    def generate_trend_report(self, results: Dict[str, Any]) -> Optional[bytes]:
        """Generate PDF report for trend analysis."""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            story = []
            
            # Title
            story.append(Paragraph("Corpus Trend Analysis Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 20))
            
            # Corpus overview
            corpus_data = results.get('corpus_data', [])
            story.append(Paragraph("Corpus Overview", self.styles['CustomHeading']))
            
            total_docs = len(corpus_data)
            total_words = sum(doc.get('word_count', 0) for doc in corpus_data)
            
            overview_info = f"""
            <b>Total Documents:</b> {total_docs}<br/>
            <b>Total Words:</b> {total_words:,}<br/>
            <b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            """
            story.append(Paragraph(overview_info, self.styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Document list
            if corpus_data:
                story.append(Paragraph("Document Details", self.styles['CustomHeading']))
                
                doc_data = [['Document', 'Date', 'Word Count']]
                for doc in corpus_data:
                    doc_data.append([
                        doc.get('filename', 'Unknown'),
                        doc.get('date', datetime.now()).strftime('%Y-%m-%d'),
                        f"{doc.get('word_count', 0):,}"
                    ])
                
                doc_table = Table(doc_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
                doc_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(doc_table)
                story.append(PageBreak())
            
            # Topic analysis
            topics = results.get('topics', [])
            if topics:
                story.append(Paragraph("Topic Analysis", self.styles['CustomHeading']))
                
                # Group topics
                topic_groups = {}
                for topic_info in topics:
                    topic_id = topic_info.get('topic_id', -1)
                    if topic_id != -1:
                        if topic_id not in topic_groups:
                            topic_groups[topic_id] = []
                        topic_groups[topic_id].append(topic_info)
                
                for topic_id, topic_docs in topic_groups.items():
                    if topic_docs:
                        story.append(Paragraph(f"Topic {topic_id}", self.styles['CustomSubHeading']))
                        
                        # Get topic words from first document in this topic
                        topic_words = topic_docs[0].get('topic_words', [])
                        if topic_words:
                            words = ', '.join([word for word, score in topic_words])
                            story.append(Paragraph(f"<b>Key Words:</b> {words}", self.styles['Normal']))
                        
                        # List documents in this topic
                        doc_names = [doc.get('filename', 'Unknown') for doc in topic_docs]
                        story.append(Paragraph(f"<b>Documents:</b> {', '.join(doc_names)}", self.styles['Normal']))
                        story.append(Spacer(1, 10))
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"Error generating trend PDF: {e}")
            return None