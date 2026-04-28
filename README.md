# CGF-FGF Lab

A minimal Streamlit app for running CGF-FGF experiments and generating compact review-ready reports with OpenAI or Claude.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## API keys

Do not commit API keys.

For private/local use, either:

- paste a temporary key into the sidebar, or
- set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` as an environment variable.

For Streamlit Community Cloud, only add a secret if you intentionally want the deployed app to use that key:

```toml
OPENAI_API_KEY = "your-api-key"
ANTHROPIC_API_KEY = "your-api-key"
```

For public review links, prefer asking each reviewer to paste their own temporary key in the sidebar.
