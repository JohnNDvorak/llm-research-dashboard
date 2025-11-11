"""Main Streamlit dashboard application for LLM Research Dashboard."""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

# Setup Python path
def ensure_path():
    """Ensure we can import our modules regardless of how we're run."""
    current_file = Path(__file__).resolve()

    # If we're in src/dashboard/, go to project root
    if current_file.parent.name == 'dashboard' and current_file.parent.parent.name == 'src':
        project_root = current_file.parent.parent.parent
    else:
        # Already at project root or similar
        project_root = current_file.parent

    # Add project root to sys.path
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    # Also add src directory if it exists
    src_dir = project_root / 'src'
    if src_dir.exists():
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

    return project_root

# Ensure paths are set
ensure_path()

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger

# Import our modules
try:
    # Try importing from project root
    from src.storage.database import PaperDatabase
    from src.embeddings.semantic_search import SemanticSearch
    from src.utils.config_loader import get_config, get_stage_keywords
    from src.utils.cost_tracker import CostTracker
except ImportError:
    # If that fails, try direct imports (if we're already in src)
    from storage.database import PaperDatabase
    from embeddings.semantic_search import SemanticSearch
    from utils.config_loader import get_config, get_stage_keywords
    from utils.cost_tracker import CostTracker


# Configure Streamlit page
st.set_page_config(
    page_title="LLM Research Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .stage-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stage-1 { background-color: #ff6b6b; color: white; }
    .stage-2 { background-color: #ffd93d; color: black; }
    .stage-3 { background-color: #6bcf7f; color: white; }
    .stage-4 { background-color: #4ecdc4; color: white; }
    .stage-5 { background-color: #45b7d1; color: white; }
    .stage-6 { background-color: #9b59b6; color: white; }
    .stage-7 { background-color: #f39c12; color: white; }
    .stage-8 { background-color: #e74c3c; color: white; }
    .paper-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #666;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'db' not in st.session_state:
        st.session_state.db = PaperDatabase()
    if 'semantic_search' not in st.session_state:
        st.session_state.semantic_search = SemanticSearch()
    if 'cost_tracker' not in st.session_state:
        st.session_state.cost_tracker = CostTracker()
    if 'selected_paper' not in st.session_state:
        st.session_state.selected_paper = None
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""


def render_sidebar():
    """Render the sidebar with navigation and filters."""
    st.sidebar.markdown("## 🧠 LLM Research Dashboard")

    # Navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["🔍 Search & Browse", "📊 Analytics", "💰 Costs", "⚙️ Settings"]
    )

    # Quick Stats
    st.sidebar.markdown("### 📈 Quick Stats")
    try:
        total_papers = st.session_state.db.get_total_papers()
        st.sidebar.metric("Total Papers", total_papers)

        # Get papers from last 7 days
        recent_papers = st.session_state.db.get_recent_papers(days=7)
        st.sidebar.metric("Last 7 Days", len(recent_papers))
    except Exception as e:
        st.sidebar.error(f"Database error: {e}")

    return page


def render_search_and_browse():
    """Render the main search and browse page."""
    st.markdown('<h1 class="main-header">🔍 Search & Browse Papers</h1>', unsafe_allow_html=True)

    # Search bar
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Search papers...",
            value=st.session_state.search_query,
            placeholder="Enter keywords, paper title, or semantic search..."
        )
    with col2:
        st.write("")
        st.write("")
        search_button = st.button("Search", type="primary")

    # Advanced filters
    with st.expander("🎯 Advanced Filters", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            # Stage filter
            stages = get_stage_keywords()
            selected_stages = st.multiselect(
                "Development Stages",
                list(stages.keys()),
                default=[]
            )

        with col2:
            # Date range
            date_range = st.date_input(
                "Date Range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                max_value=datetime.now()
            )

        with col3:
            # Source filter
            sources = ["arXiv", "Twitter/X", "LinkedIn"]
            selected_sources = st.multiselect(
                "Sources",
                sources,
                default=sources
            )

    # Perform search
    papers = []
    if search_query or search_button or selected_stages or selected_sources:
        try:
            # Try semantic search first
            if search_query:
                papers = st.session_state.semantic_search.search(
                    query=search_query,
                    limit=20,
                    stages=selected_stages if selected_stages else None,
                    date_from=date_range[0] if date_range else None,
                    date_to=date_range[1] if date_range else None
                )
            else:
                # Fallback to database filtering
                papers = st.session_state.db.get_papers(
                    stages=selected_stages if selected_stages else None,
                    date_from=date_range[0] if date_range else None,
                    date_to=date_range[1] if date_range else None,
                    limit=50
                )

            # Filter by source if specified
            if selected_sources:
                source_map = {
                    "arXiv": "arxiv_id",
                    "Twitter/X": "twitter_url",
                    "LinkedIn": "linkedin_url"
                }
                filtered_papers = []
                for source in selected_sources:
                    field = source_map[source]
                    filtered_papers.extend([p for p in papers if p.get(field)])
                papers = filtered_papers

        except Exception as e:
            st.error(f"Search error: {e}")
            papers = []

    # Display results
    if papers:
        st.success(f"Found {len(papers)} papers")

        for paper in papers:
            render_paper_card(paper)
    else:
        st.info("No papers found. Try adjusting your search or filters.")

        # Show recent papers if no search
        st.markdown("### 📚 Recent Papers")
        try:
            recent_papers = st.session_state.db.get_recent_papers(days=7, limit=5)
            for paper in recent_papers:
                render_paper_card(paper)
        except Exception as e:
            st.error(f"Error loading recent papers: {e}")


def render_paper_card(paper):
    """Render a single paper card."""
    # Paper container
    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            # Title and link
            title_html = paper['title']
            if paper.get('arxiv_id'):
                title_html = f"[{title_html}](https://arxiv.org/abs/{paper['arxiv_id']})"
            elif paper.get('url'):
                title_html = f"[{title_html}]({paper['url']})"

            st.markdown(f"### {title_html}")

            # Authors
            authors = paper.get('authors', [])
            if authors:
                author_text = ', '.join(authors[:3])
                if len(authors) > 3:
                    author_text += f" and {len(authors) - 3} others"
                st.markdown(f"*Authors: {author_text}*")

            # Summary
            summary = paper.get('summary', paper.get('abstract', ''))[:300]
            if len(summary) == 300:
                summary += "..."
            st.markdown(summary)

            # Stages
            stages = paper.get('stages', [])
            if stages:
                stage_html = ""
                for stage in stages:
                    stage_num = stage.split(':')[0].replace('Stage ', '')
                    stage_html += f'<span class="stage-badge stage-{stage_num}">{stage}</span> '
                st.markdown(stage_html, unsafe_allow_html=True)

            # Metadata
            meta_cols = st.columns(4)
            with meta_cols[0]:
                if paper.get('arxiv_id'):
                    st.caption(f"📄 arXiv:{paper['arxiv_id']}")
            with meta_cols[1]:
                if paper.get('twitter_url'):
                    st.caption("🐦 Twitter")
            with meta_cols[2]:
                if paper.get('linkedin_url'):
                    st.caption("💼 LinkedIn")
            with meta_cols[3]:
                if paper.get('published_date'):
                    st.caption(f"📅 {paper['published_date']}")

        with col2:
            # View details button
            if st.button("View Details", key=f"view_{paper['id']}"):
                st.session_state.selected_paper = paper
                st.rerun()

            # Relevance score if from semantic search
            if 'relevance_score' in paper:
                st.markdown(f"**Relevance:** {paper['relevance_score']:.2f}")

        st.markdown("---")


def render_paper_details():
    """Render detailed view of a selected paper."""
    if not st.session_state.selected_paper:
        return

    paper = st.session_state.selected_paper

    st.markdown("## 📄 Paper Details")

    # Back button
    if st.button("← Back to Results"):
        st.session_state.selected_paper = None
        st.rerun()

    # Title and metadata
    st.markdown(f"### {paper['title']}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Authors:** {', '.join(paper.get('authors', []))}")
        if paper.get('published_date'):
            st.markdown(f"**Published:** {paper['published_date']}")
    with col2:
        if paper.get('arxiv_id'):
            st.markdown(f"**arXiv ID:** [{paper['arxiv_id']}](https://arxiv.org/abs/{paper['arxiv_id']})")
        if paper.get('url'):
            st.markdown(f"**URL:** [Link]({paper['url']})")

    # Abstract
    if paper.get('abstract'):
        st.markdown("### Abstract")
        st.markdown(paper['abstract'])

    # Analysis results
    if paper.get('stages'):
        st.markdown("### 🎯 Development Stage Analysis")
        for stage in paper['stages']:
            st.markdown(f"- {stage}")

    if paper.get('summary'):
        st.markdown("### 📝 Summary")
        st.markdown(paper['summary'])

    if paper.get('key_insights'):
        st.markdown("### 💡 Key Insights")
        for insight in paper['key_insights']:
            st.markdown(f"- {insight}")

    # Social metrics
    if paper.get('twitter_metrics') or paper.get('linkedin_metrics'):
        st.markdown("### 📊 Social Metrics")

        if paper.get('twitter_metrics'):
            metrics = paper['twitter_metrics']
            st.markdown("**Twitter/X:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Likes", metrics.get('likes', 0))
            with col2:
                st.metric("Retweets", metrics.get('retweets', 0))
            with col3:
                st.metric("Replies", metrics.get('replies', 0))
            with col4:
                st.metric("Views", metrics.get('views', 0))

        if paper.get('linkedin_metrics'):
            metrics = paper['linkedin_metrics']
            st.markdown("**LinkedIn:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Likes", metrics.get('likes', 0))
            with col2:
                st.metric("Comments", metrics.get('comments', 0))
            with col3:
                st.metric("Shares", metrics.get('shares', 0))

    # Similar papers
    if paper.get('title') or paper.get('abstract'):
        st.markdown("### 🔍 Similar Papers")
        try:
            query = paper.get('title', '')[:100]
            similar = st.session_state.semantic_search.search(
                query=query,
                limit=5,
                exclude_ids=[paper['id']]
            )

            for sim_paper in similar:
                with st.expander(sim_paper['title']):
                    st.markdown(sim_paper.get('summary', sim_paper.get('abstract', ''))[:200] + "...")
                    if st.button("View", key=f"similar_{sim_paper['id']}"):
                        st.session_state.selected_paper = sim_paper
                        st.rerun()
        except Exception as e:
            st.error(f"Error finding similar papers: {e}")


def render_analytics():
    """Render the analytics dashboard."""
    st.markdown('<h1 class="main-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)

    # Time range selector
    col1, col2, col3 = st.columns(3)
    with col1:
        time_range = st.selectbox(
            "Time Range",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
            index=1
        )

    # Map time range to days
    days_map = {
        "Last 7 days": 7,
        "Last 30 days": 30,
        "Last 90 days": 90,
        "All time": None
    }
    days = days_map[time_range]

    # Get analytics data
    try:
        # Papers over time
        papers_over_time = st.session_state.db.get_papers_over_time(days=days)

        if papers_over_time:
            df_time = pd.DataFrame(papers_over_time)
            df_time['date'] = pd.to_datetime(df_time['date'])

            fig = px.line(
                df_time,
                x='date',
                y='count',
                title=f'Papers Added Over Time ({time_range})',
                labels={'count': 'Number of Papers', 'date': 'Date'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Stage distribution
        stage_dist = st.session_state.db.get_stage_distribution(days=days)

        if stage_dist:
            col1, col2 = st.columns(2)

            with col1:
                # Pie chart
                stages = [s['stage'] for s in stage_dist]
                counts = [s['count'] for s in stage_dist]

                fig = px.pie(
                    values=counts,
                    names=stages,
                    title='Papers by Development Stage'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Bar chart
                df_stages = pd.DataFrame(stage_dist)
                fig = px.bar(
                    df_stages,
                    x='stage',
                    y='count',
                    title='Papers by Development Stage',
                    labels={'count': 'Number of Papers', 'stage': 'Stage'}
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

        # Top sources
        source_stats = st.session_state.db.get_source_statistics(days=days)

        if source_stats:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### 📄 arXiv")
                st.metric("Papers", source_stats.get('arxiv', 0))

            with col2:
                st.markdown("### 🐦 Twitter/X")
                st.metric("Papers", source_stats.get('twitter', 0))

            with col3:
                st.markdown("### 💼 LinkedIn")
                st.metric("Papers", source_stats.get('linkedin', 0))

        # Most discussed papers
        most_discussed = st.session_state.db.get_most_discussed_papers(days=days, limit=5)

        if most_discussed:
            st.markdown("### 🔥 Most Discussed Papers")

            for paper in most_discussed:
                score = 0
                if paper.get('twitter_metrics'):
                    score += paper['twitter_metrics'].get('likes', 0) * 1
                    score += paper['twitter_metrics'].get('retweets', 0) * 2
                    score += paper['twitter_metrics'].get('replies', 0) * 1.5
                if paper.get('linkedin_metrics'):
                    score += paper['linkedin_metrics'].get('likes', 0) * 1
                    score += paper['linkedin_metrics'].get('comments', 0) * 3
                    score += paper['linkedin_metrics'].get('shares', 0) * 2

                with st.expander(f"{paper['title']} (Score: {score})"):
                    st.markdown(f"**Authors:** {', '.join(paper.get('authors', []))}")
                    if paper.get('twitter_metrics'):
                        st.write(f"Twitter: {paper['twitter_metrics']}")
                    if paper.get('linkedin_metrics'):
                        st.write(f"LinkedIn: {paper['linkedin_metrics']}")

    except Exception as e:
        st.error(f"Error loading analytics: {e}")


def render_costs():
    """Render the cost tracking dashboard."""
    st.markdown('<h1 class="main-header">💰 Cost Tracking</h1>', unsafe_allow_html=True)

    try:
        # Get cost data
        costs = st.session_state.cost_tracker.get_total_costs()
        daily_costs = st.session_state.cost_tracker.get_daily_costs(days=30)
        provider_costs = st.session_state.cost_tracker.get_costs_by_provider()

        # Summary cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">${costs.get("total", 0):.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Total Spent</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">${costs.get("llm", 0):.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">LLM APIs</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">${costs.get("embeddings", 0):.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Embeddings</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            daily_avg = costs.get("total", 0) / 30 if costs.get("total", 0) > 0 else 0
            st.markdown(f'<div class="metric-value">${daily_avg:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Daily Avg (30d)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Daily costs chart
        if daily_costs:
            st.markdown("### 📈 Daily Costs (Last 30 Days)")

            df_daily = pd.DataFrame(daily_costs)
            df_daily['date'] = pd.to_datetime(df_daily['date'])

            fig = px.line(
                df_daily,
                x='date',
                y='cost',
                title='Daily API Costs',
                labels={'cost': 'Cost ($)', 'date': 'Date'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Provider breakdown
        if provider_costs:
            st.markdown("### 🏢 Costs by Provider")

            df_provider = pd.DataFrame([
                {"Provider": "xAI", "Cost": costs.get("xai", 0)},
                {"Provider": "OpenAI", "Cost": costs.get("openai", 0)},
                {"Provider": "Together AI", "Cost": costs.get("together", 0)},
                {"Provider": "Other", "Cost": costs.get("other", 0)}
            ])

            fig = px.bar(
                df_provider,
                x='Provider',
                y='Cost',
                title='Cost Breakdown by Provider'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Budget status
        st.markdown("### 📊 Budget Status")

        budget_config = get_config("budget_modes.yaml")
        current_mode = os.getenv("BUDGET_MODE", "moderate")
        budget = budget_config.get(current_mode, {})

        daily_budget = budget.get("daily_limit", 10.0)
        monthly_budget = budget.get("monthly_limit", 300.0)

        col1, col2 = st.columns(2)

        with col1:
            # Daily budget
            today_spent = costs.get("today", 0)
            remaining = daily_budget - today_spent

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=today_spent,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Daily Budget (${daily_budget})"},
                delta={'reference': daily_budget},
                gauge={
                    'axis': {'range': [None, daily_budget]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, daily_budget * 0.5], 'color': "lightgray"},
                        {'range': [daily_budget * 0.5, daily_budget * 0.8], 'color': "yellow"},
                        {'range': [daily_budget * 0.8, daily_budget], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': daily_budget * 0.9
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Monthly budget
            monthly_spent = costs.get("this_month", costs.get("total", 0))
            remaining_monthly = monthly_budget - monthly_spent

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=monthly_spent,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Monthly Budget (${monthly_budget})"},
                delta={'reference': monthly_budget},
                gauge={
                    'axis': {'range': [None, monthly_budget]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, monthly_budget * 0.5], 'color': "lightgray"},
                        {'range': [monthly_budget * 0.5, monthly_budget * 0.8], 'color': "lightgreen"},
                        {'range': [monthly_budget * 0.8, monthly_budget], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': monthly_budget * 0.9
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        # Recent transactions
        st.markdown("### 📋 Recent API Calls")

        recent_transactions = st.session_state.cost_tracker.get_recent_transactions(limit=20)

        if recent_transactions:
            df_tx = pd.DataFrame(recent_transactions)

            # Format for display
            df_tx['timestamp'] = pd.to_datetime(df_tx['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            df_tx['cost'] = df_tx['cost'].round(4)

            st.dataframe(
                df_tx[['timestamp', 'provider', 'model', 'operation', 'tokens', 'cost']],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No recent API calls found.")

    except Exception as e:
        st.error(f"Error loading cost data: {e}")


def render_settings():
    """Render the settings page."""
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)

    st.markdown("### Configuration")
    st.info("Configuration is managed through YAML files in the `/config` directory and environment variables.")

    # Show current configuration
    st.markdown("#### Current Settings")

    # LLM Configuration
    llm_config = get_config("llm_config.yaml")
    st.markdown("##### LLM Providers")
    st.json({
        "primary_provider": llm_config.get("primary_provider"),
        "primary_model": llm_config.get("primary_model"),
        "fallback_enabled": llm_config.get("fallback_enabled", True)
    })

    # Budget Mode
    budget_mode = os.getenv("BUDGET_MODE", "moderate")
    st.markdown(f"##### Budget Mode: `{budget_mode}`")

    # API Keys (masked)
    st.markdown("##### API Keys Status")

    api_keys = {
        "XAI_API_KEY": "✅ Set" if os.getenv("XAI_API_KEY") else "❌ Not set",
        "OPENAI_API_KEY": "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not set",
        "TOGETHER_API_KEY": "✅ Set" if os.getenv("TOGETHER_API_KEY") else "❌ Not set",
        "TWITTER_BEARER_TOKEN": "✅ Set" if os.getenv("TWITTER_BEARER_TOKEN") else "❌ Not set",
        "LINKEDIN_EMAIL": "✅ Set" if os.getenv("LINKEDIN_EMAIL") else "❌ Not set"
    }

    for key, status in api_keys.items():
        st.markdown(f"- **{key}**: {status}")

    # Database info
    st.markdown("##### Database Information")
    try:
        db_path = os.getenv("DB_PATH", "data/papers.db")
        st.markdown(f"- **Database Path**: `{db_path}`")

        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            st.markdown(f"- **Database Size**: {size_mb:.2f} MB")

        total_papers = st.session_state.db.get_total_papers()
        st.markdown(f"- **Total Papers**: {total_papers}")

        # ChromaDB info
        chroma_path = "data/chroma"
        if os.path.exists(chroma_path):
            st.markdown(f"- **ChromaDB Path**: `{chroma_path}`")
    except Exception as e:
        st.error(f"Error getting database info: {e}")

    # Actions
    st.markdown("### Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh Data", type="secondary"):
            st.rerun()

    with col2:
        if st.button("📊 Run Analysis", type="primary"):
            st.info("Analysis feature coming soon!")

    with col3:
        if st.button("🗑️ Clear Cache", type="secondary"):
            if 'semantic_search' in st.session_state:
                # Clear caches
                if hasattr(st.session_state.semantic_search, 'cache'):
                    st.session_state.semantic_search.cache.clear()
            st.success("Cache cleared!")
            st.rerun()


def main():
    """Main dashboard entry point."""
    # Initialize session state
    init_session_state()

    # Render sidebar
    page = render_sidebar()

    # Render selected page
    if st.session_state.selected_paper:
        render_paper_details()
    elif page == "🔍 Search & Browse":
        render_search_and_browse()
    elif page == "📊 Analytics":
        render_analytics()
    elif page == "💰 Costs":
        render_costs()
    elif page == "⚙️ Settings":
        render_settings()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.875rem;'>"
        "LLM Research Dashboard - Built with Streamlit • "
        f"<a href='https://github.com/yourusername/llm-research-dashboard' target='_blank'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()