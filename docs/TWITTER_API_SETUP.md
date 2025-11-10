# X (formerly Twitter) API Setup Guide

## API Tiers and Limitations

### Free Tier (Recommended for testing)
- **Cost**: Free
- **Requests**: 500,000 tweets/month
- **Search**: Last 7 days only
- **Rate limit**: 300 requests/15 minutes

### Basic Tier ($100/month)
- **Requests**: 2 million tweets/month
- **Search**: Last 30 days
- **Higher rate limits**

### Pro Tier ($5,000/month)
- **Full archive search**
- **Real-time streaming**
- **Advanced filtering

## Recommended Configuration for LLM Research Dashboard

### Using Free Tier
The free tier is sufficient for research purposes:
- Track 12-16 accounts
- Search 9 hashtags
- Fetch up to 1,000 posts/day
- Stay within 500K monthly limit

## Rate Limiting Best Practices

1. **Delays Between Requests**
   ```python
   # Current configuration (2 seconds)
   rate_limit_delay: 2
   ```

2. **Batch Processing**
   - Fetch multiple tweets per request
   - Use `max_results=100` where possible

3. **Monitoring**
   - Check rate limit headers
   - Implement backoff on errors

## Security Notes

1. **Never commit API keys** to git
2. **Use environment variables** only
3. **Rotate tokens** if compromised
4. **Monitor usage** to avoid overages

## Troubleshooting

### Common Errors

1. **401 Unauthorized**
   - Check Bearer Token is correct
   - Ensure app has proper permissions

2. **429 Rate Limited**
   - Increase `rate_limit_delay` in config
   - Wait for rate limit window to reset

3. **403 Forbidden**
   - Check app is approved
   - Verify API terms compliance

### Testing Your Setup

```python
from src.fetch.twitter_fetcher import TwitterFetcher

# Test basic initialization
try:
    fetcher = TwitterFetcher()
    print("✅ X API connection successful")

    # Test a small fetch
    papers = fetcher.fetch_from_accounts(days=1)
    print(f"✅ Fetched {len(papers)} papers")

except Exception as e:
    print(f"❌ Error: {e}")
```

## Next Steps

1. Get your X Bearer Token
2. Add it to `.env` file
3. Test with the code above
4. Adjust `config/queries.yaml` if needed
5. Monitor your usage in X Developer Dashboard