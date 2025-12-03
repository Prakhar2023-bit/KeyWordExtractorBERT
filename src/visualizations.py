import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import numpy as np
import seaborn as sns
from collections import Counter


# =========================
# ADVANCED VISUALIZATIONS
# =========================

def create_bar_chart(df: pd.DataFrame, theme_text_color: str):
    """Creates an advanced horizontal bar chart for keywords."""
    if df is None or (hasattr(df, 'empty') and df.empty):
        return go.Figure()

    # Handle both "Relevance" and "Relevance Score" column names
    score_col = "Relevance Score" if "Relevance Score" in df.columns else "Relevance"
    
    df_top = df.head(25).iloc[::-1]  # reverse so top is at top
    fig = px.bar(
        df_top,
        x=score_col,
        y="Keyword",
        orientation='h',
        title="Top Keywords by Relevance",
        color=score_col,
        color_continuous_scale="Viridis",
        text=score_col
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=theme_text_color,
        xaxis_title="Relevance Score",
        yaxis_title="Keyword",
        margin=dict(l=10, r=10, t=40, b=40),
        height=600
    )
    return fig


def create_wordcloud(keywords_scores: list, theme: str = "dark"):
    """Creates a detailed WordCloud visualization."""
    if not keywords_scores:
        return None

    background = "black" if theme == "dark" else "white"

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color=background,
        colormap='viridis',
        contour_color='steelblue',
        contour_width=1.5
    ).generate_from_frequencies(dict(keywords_scores))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def render_knowledge_graph(G: nx.Graph):
    """Renders a NetworkX graph as an advanced interactive PyVis visualization."""
    if not G.nodes:
        st.info("No entities or relationships were found to build a graph.")
        return

    # Compute node importance
    centrality = nx.degree_centrality(G)

    net = Network(
        height='650px',
        width='100%',
        bgcolor='#111111',
        font_color='white',
        notebook=True,
        cdn_resources='in_line'
    )

    for node, data in G.nodes(data=True):
        node_type = data.get("type", "Unknown")
        size = 15 + centrality.get(node, 0) * 100
        color = {
            "PERSON": "#1f77b4",
            "ORG": "#ff7f0e",
            "GPE": "#2ca02c",
            "NOUN_CHUNK": "#9467bd"
        }.get(node_type, "#8c564b")

        net.add_node(node, label=node, size=size, color=color, title=node_type)

    for src, dst, data in G.edges(data=True):
        weight = data.get("weight", 1)
        net.add_edge(src, dst, value=weight, title=f"Weight: {weight}")

    net.set_options("""
    var options = {
      "physics": { 
        "barnesHut": { 
          "gravitationalConstant": -20000, 
          "centralGravity": 0.3, 
          "springLength": 120 
        },
        "minVelocity": 0.75
      },
      "edges": {
        "color": {"inherit": "from"},
        "smooth": {"enabled": true, "type": "continuous"}
      }
    }
    """)
    html_content = net.generate_html()
    components.html(html_content, height=680)


def create_sentiment_chart(sentiment_scores: dict, theme_text_color: str):
    """Creates an advanced Plotly pie/donut chart for sentiment distribution."""
    if not sentiment_scores:
        return go.Figure()

    labels = list(sentiment_scores.keys())
    values = list(sentiment_scores.values())

    colors = {
        'Positive': '#22c55e',  # green
        'Neutral': '#6b7280',   # gray
        'Negative': '#ef4444'   # red
    }

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker_colors=[colors.get(label, '#cccccc') for label in labels],
        textinfo='label+percent',
        insidetextorientation='radial'
    )])

    fig.update_layout(
        title_text="Sentiment Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=theme_text_color,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def create_sentiment_trend_chart(df_sentiment: pd.DataFrame, theme_text_color: str = "white"):
    """Creates an advanced Plotly line chart for sentiment trends over time."""
    if df_sentiment is None or (hasattr(df_sentiment, 'empty') and df_sentiment.empty):
        return go.Figure()

    # Handle different column names that might be passed
    if 'Date' in df_sentiment.columns and 'Score' in df_sentiment.columns and 'Sentiment' in df_sentiment.columns:
        # For trend analysis format
        fig = px.line(
            df_sentiment, 
            x='Date', 
            y='Score', 
            color='Sentiment',
            title='Sentiment Trends Over Time',
            hover_data=['Document'] if 'Document' in df_sentiment.columns else None
        )
    elif 'timestamp' in df_sentiment.columns and 'sentiment_score' in df_sentiment.columns:
        # For time series format
        df_sorted = df_sentiment.sort_values('timestamp')
        # Optional smoothing (rolling average)
        df_sorted['rolling_sentiment'] = df_sorted['sentiment_score'].rolling(window=3, min_periods=1).mean()

        fig = go.Figure()

        # Raw sentiment points
        fig.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['sentiment_score'],
            mode='markers+lines',
            name='Daily Sentiment',
            line=dict(color='#60a5fa', width=2),
            marker=dict(size=8)
        ))

        # Smoothed trend line
        fig.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['rolling_sentiment'],
            mode='lines',
            name='Smoothed Trend',
            line=dict(color='#2563eb', width=3, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(37,99,235,0.1)'
        ))

        fig.update_layout(
            title="Average Sentiment Over Time",
            yaxis_title="Sentiment (Negative -1 → Positive +1)",
        )
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
    else:
        # Return empty figure if format not recognized
        return go.Figure()

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=theme_text_color,
        hovermode="x unified",
        height=500,
        margin=dict(l=40, r=20, t=60, b=40)
    )

    return fig


def create_entity_chart(entities, title: str = "Named Entities by Type"):
    """Creates a bar chart of entity counts with advanced styling."""
    # Handle DataFrame boolean evaluation issue
    if entities is None or (hasattr(entities, 'empty') and entities.empty) or (isinstance(entities, list) and len(entities) == 0):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No entities found', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig

    # Handle both list of dicts and DataFrame formats
    if isinstance(entities, list):
        # Count entity labels
        counts = Counter(ent.get('label', 'Unknown') for ent in entities if isinstance(ent, dict))
    else:
        # Assume DataFrame
        counts = Counter(entities['label'] if 'label' in entities.columns else [])

    if not counts:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No valid entity data', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig

    labels = list(counts.keys())
    values = list(counts.values())

    # Create Plotly figure with proper color handling
    fig = go.Figure(data=[go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(
            colorscale='viridis',  # Use colorscale instead of color
            color=values,  # Color bars by their values
            colorbar=dict(title="Count")
        )
    )])

    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis_title="Count",
        yaxis_title="Entity Type",
        height=400,
        margin=dict(l=100, r=20, t=60, b=40)
    )

    return fig

def create_topic_wordcloud(topic_words, title: str = "Topic WordCloud"):
    """Creates a WordCloud from topic words with advanced styling."""
    if not topic_words:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'No words available for {title}', 
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig

    # Handle different input formats
    if isinstance(topic_words, list):
        if topic_words and isinstance(topic_words[0], tuple):
            # List of tuples (word, score)
            topic_dict = {word: score for word, score in topic_words}
        else:
            # List of words
            topic_dict = {word: 1 for word in topic_words}
    elif isinstance(topic_words, dict):
        topic_dict = topic_words
    else:
        # Try to convert to string representation
        topic_dict = {str(topic_words): 1}

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="black",
        colormap="viridis",
        contour_color='steelblue',
        contour_width=1.5
    ).generate_from_frequencies(topic_dict)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, color='white', fontsize=16, pad=20)
    fig.patch.set_facecolor('black')
    plt.tight_layout(pad=0)
    return fig


def create_comparison_heatmap(df, title="Comparison Heatmap", cmap="viridis"):
    """Creates a heatmap for comparing values across categories."""
    if df is None or (hasattr(df, 'empty') and df.empty):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    sns.heatmap(df, annot=True, fmt=".2f", cmap=cmap, ax=ax, cbar=True)
    ax.set_title(title, color='white', fontsize=14, pad=20)
    ax.tick_params(colors='white')
    plt.tight_layout()
    return fig


def create_entity_timeline(df_entities, theme_text_color="white"):
    """Creates a timeline chart showing entity frequency over time."""
    if df_entities is None or (hasattr(df_entities, 'empty') and df_entities.empty):
        return go.Figure()

    # Count entity frequency per timestamp
    df_counts = df_entities.groupby(["timestamp", "entity"]).size().reset_index(name="count")

    fig = px.line(
        df_counts,
        x="timestamp",
        y="count",
        color="entity",
        markers=True,
        title="Entity Timeline"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=theme_text_color,
        xaxis_title="Time",
        yaxis_title="Mentions",
        hovermode="x unified",
        height=500
    )
    return fig