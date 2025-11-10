-- Initial database schema for LLM Research Dashboard
-- See PROJECT_PLAN.md for detailed schema documentation

-- Main papers table
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT,  -- JSON array
    abstract TEXT NOT NULL,
    url TEXT,
    pdf_url TEXT,

    -- Source tracking
    source TEXT,  -- 'arxiv', 'twitter', 'linkedin'
    fetch_date DATE,
    published_date DATE,

    -- arXiv specific
    arxiv_id TEXT UNIQUE,
    arxiv_category TEXT,

    -- Social metrics
    social_score INTEGER DEFAULT 0,  -- Twitter: likes + retweets

    -- X/Twitter metrics
    tweet_id TEXT UNIQUE,
    twitter_likes INTEGER DEFAULT 0,
    twitter_retweets INTEGER DEFAULT 0,
    twitter_replies INTEGER DEFAULT 0,
    twitter_poster TEXT,
    twitter_post_date DATE,

    -- LinkedIn metrics
    linkedin_post_id TEXT UNIQUE,
    linkedin_post_url TEXT,
    linkedin_author_name TEXT,
    linkedin_author_title TEXT,
    linkedin_company TEXT,
    linkedin_likes INTEGER DEFAULT 0,
    linkedin_comments INTEGER DEFAULT 0,
    linkedin_shares INTEGER DEFAULT 0,
    linkedin_views INTEGER DEFAULT 0,
    linkedin_post_date DATE,
    professional_score INTEGER DEFAULT 0,

    -- Combined scoring (calculated by PaperDeduplicator)
    combined_score FLOAT,  -- Formula: (social*0.4) + (prof*0.6) + (recency*0.3)

    -- Analysis results
    analyzed BOOLEAN DEFAULT 0,
    stages TEXT,  -- JSON array of assigned stages
    summary TEXT,
    key_insights TEXT,  -- JSON array
    metrics TEXT,  -- JSON: extracted performance gains
    complexity_score FLOAT,

    -- LLM tracking
    model_used TEXT,
    analysis_cost FLOAT,

    -- Vector embeddings
    embedding_generated BOOLEAN DEFAULT 0,
    embedding_model TEXT,
    embedding_cost FLOAT,
    chroma_id TEXT,  -- ID in ChromaDB

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cost tracking table
CREATE TABLE IF NOT EXISTS cost_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT,
    model TEXT,
    paper_id TEXT,
    operation_type TEXT,  -- 'analysis' or 'embedding'
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);


-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_papers_stages ON papers(stages);
CREATE INDEX IF NOT EXISTS idx_papers_fetch_date ON papers(fetch_date);
CREATE INDEX IF NOT EXISTS idx_papers_analyzed ON papers(analyzed);
CREATE INDEX IF NOT EXISTS idx_papers_social_score ON papers(social_score);
CREATE INDEX IF NOT EXISTS idx_papers_professional_score ON papers(professional_score);
CREATE INDEX IF NOT EXISTS idx_papers_combined_score ON papers(combined_score);
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
CREATE INDEX IF NOT EXISTS idx_papers_chroma_id ON papers(chroma_id);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_provider ON cost_tracking(provider);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_timestamp ON cost_tracking(timestamp);

-- Additional indices for new columns
CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON papers(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_tweet_id ON papers(tweet_id);
CREATE INDEX IF NOT EXISTS idx_papers_linkedin_post_id ON papers(linkedin_post_id);
CREATE INDEX IF NOT EXISTS idx_papers_published_date ON papers(published_date);
