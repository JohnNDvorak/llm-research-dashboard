# LLM Research Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/yourusername/llm-research-dashboard)

Automated system for discovering, analyzing, and organizing LLM research papers with semantic search.

## 🚀 Live Demo

Visit the dashboard at: [https://your-app.streamlit.app](https://your-app.streamlit.app)

## ✨ Features

- **🔍 Semantic Search**: Find papers by meaning, not just keywords
- **📊 Analytics Dashboard**: Track trends and insights
- **💰 Cost Tracking**: Monitor API spending in real-time
- **🎯 Stage Classification**: Papers categorized into 8 development stages
- **📱 Multi-Source**: arXiv, Twitter/X, LinkedIn integration
- **🔧 Free Hosting**: Deploy on Streamlit Cloud ($0/month)

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.11+
- **Database**: SQLite + ChromaDB (vector search)
- **LLM**: xAI grok-4-fast-reasoning
- **Embeddings**: OpenAI text-embedding-3-small
- **Deployment**: Streamlit Cloud (FREE tier)

## 📋 Quick Start

### Prerequisites
- Python 3.11 or higher
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/llm-research-dashboard
   cd llm-research-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the dashboard**
   ```bash
   streamlit run src/dashboard/app.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:8501`

## 🔑 Environment Variables

Create a `.env` file with:

```bash
# Required for full functionality
XAI_API_KEY=your_xai_key_here          # Primary LLM
OPENAI_API_KEY=your_openai_key_here      # Embeddings
TWITTER_BEARER_TOKEN=your_twitter_token  # Social metrics
LINKEDIN_EMAIL=your@email.com           # Professional metrics

# Optional fallbacks
TOGETHER_API_KEY=your_together_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

## 📊 Usage

### Searching Papers
- Use semantic search to find papers by concept
- Filter by development stage, date, or source
- Click "View Details" for full analysis

### Analytics
- View paper trends over time
- See distribution by development stage
- Track most discussed papers

### Cost Monitoring
- Real-time API cost tracking
- Daily and monthly budget gauges
- Detailed transaction history

## 💡 Deployment

### Deploy to Streamlit Cloud (FREE)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Connect Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Select the repository and branch
   - Main file path: `src/dashboard/app.py`

3. **Set Secrets**
   In Streamlit Cloud dashboard, add your environment variables as secrets.

4. **Deploy!**
   Click "Deploy" and your app will be live at `https://your-app.streamlit.app`

### Self-Hosting (Optional)

For more control, deploy to your own server:

```bash
# Using Docker (recommended)
docker build -t llm-dashboard .
docker run -p 8501:8501 llm-dashboard

# Or directly
pip install -r requirements.txt
streamlit run src/dashboard/app.py --server.address 0.0.0.0
```

## 📈 Scaling Guide

### Free Tier (Streamlit Cloud)
- ✅ Unlimited public users
- ✅ Custom subdomain
- ✅ SSL certificate
- ✅ No credit card required

### When to Upgrade
- Page load > 3 seconds
- >50 concurrent users
- Need background jobs

### Self-Hosted VPS (~$6/month)
- DigitalOcean 2GB RAM
- Full control over resources
- Custom domains free
- Background processing support

## 🔧 Configuration

Configuration is managed through YAML files in `/config`:

- `stages.yaml` - Development stage definitions
- `llm_config.yaml` - LLM provider settings
- `queries.yaml` - Search queries for data sources
- `budget_modes.yaml` - Cost control settings

## 📝 Project Structure

```
llm-research-dashboard/
├── src/
│   ├── dashboard/          # Streamlit UI
│   ├── llm/               # LLM providers
│   ├── analysis/          # Paper analysis
│   ├── embeddings/        # Vector search
│   ├── fetch/            # Data sources
│   ├── storage/          # Database layer
│   └── utils/            # Utilities
├── config/               # Configuration files
├── tests/               # Test suite
├── data/                # Database storage
└── scripts/             # Helper scripts
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/test_dashboard.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing framework
- [xAI](https://x.ai/) for Grok model access
- [ChromaDB](https://www.trychroma.com/) for vector database
- All contributors and users of this dashboard

## 📞 Support

- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/llm-research-dashboard/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/llm-research-dashboard/discussions)

---

**Built with ❤️ for the LLM research community**