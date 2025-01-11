# FastAsk

A simple application that uses parallel LLM calls to gather context from a large
amount of text and then use it to chat. I am currently using deepseek. Use gitingest.com to copy tokens from public sources. I use this for documentation.

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file in the project root and add your OpenRouter API key:
```
OPENROUTER_API_KEY=your_api_key_here
```

4. Run the application:
```bash
streamlit run app.py
```

