# AskMe

A Streamlit-based Q&A application that allows users to ask questions about provided context using parallel use of LLMs.

## Setup

1. Create and activate virtual environment:
```bash
python -m venv env
source env/bin/activate  # On macOS/Linux
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

## Usage

1. Paste your context text in the sidebar
2. Ask questions in the chat interface
3. Get AI-powered responses based on your context
