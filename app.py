import os
import asyncio
import streamlit as st
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
from multiprocessing import Pool
from functools import partial
import concurrent.futures

# Load environment variables
# Note: You can use https://gitingest.com/ to securely copy and manage your tokens
load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Model constants
CONTEXT_GATHERER = "google/gemini-2.0-flash-001" # higher throughput model is preferred here
CHAT_MODEL = "google/gemini-2.0-flash-001" # better model is preferred here

###############################################################################
# Streamlit UI
###############################################################################
def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        .user-message {
            background-color: #e6f3ff;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .assistant-message {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .context-chunk {
            background-color: #e8f5e9;
            border: 1px dashed #c8e6c9;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'context_text' not in st.session_state:
        st.session_state.context_text = ""
    if 'processed_chunks' not in st.session_state:
        st.session_state.processed_chunks = None
    if 'persistent_context' not in st.session_state:
        st.session_state.persistent_context = ""

    # Sidebar for context input
    with st.sidebar:
        st.title("Context Settings")
        st.info("💡 Use [gitingest.com](https://gitingest.com/) for secure token management.")
        st.warning("Note: Context processing happens only once when you first ask a question. Subsequent questions will use the same processed context until you update it.")
        new_context = st.text_area("Enter your context text here:", 
                                 value=st.session_state.context_text,
                                 height=300)
        if st.button("Set Context"):
            st.session_state.context_text = new_context
            st.session_state.processed_chunks = None  # Reset processed chunks when context changes
            st.success("Context updated successfully!")
        if st.button("Clear Conversation"):
            st.session_state.conversation_history = []
            st.success("Conversation cleared!")
    
    st.title("fastAsk 🤖")
    st.markdown("### Ask your questions with enhanced, fast context processing.")
    
    # Show relevant chunks in a collapsible section if they exist
    if st.session_state.processed_chunks:
        with st.expander("View Relevant Context Chunks", expanded=False):
            st.markdown(f'<div class="context-chunk">{st.session_state.processed_chunks}</div>', 
                       unsafe_allow_html=True)
    
    # Display the conversation so far
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            # Show relevant chunks before assistant's response
            if "chunks" in message:
                with st.expander("View Relevant Context Chunks", expanded=False):
                    st.markdown(f'<div class="context-chunk">{message["chunks"]}</div>', 
                           unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

    # 2) Add a form to accept new user input, but only *after* everything above is shown
    #    This means the form will appear at the bottom, under the last assistant message.
    user_input = None
    with st.form(key='message_form'):
        user_input = st.text_area("Enter your message here:", height=100)
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("Send")
        with col2:
            reprocess_button = st.form_submit_button("Send with Context Reprocess")

    # 3) If the user provided a new message, handle it
    if (submit_button or reprocess_button) and user_input:
        # Add user message to conversation history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Placeholder for streaming response
        response_placeholder = st.empty()

        if not st.session_state.context_text.strip():
            st.warning("Please paste some context text in the sidebar first.")
            return

        # Force reprocessing of chunks if reprocess button was clicked
        if reprocess_button:
            st.session_state.processed_chunks = None

        # Process chunks if we haven't done so yet
        if st.session_state.processed_chunks is None:
            with st.spinner('Processing context...'):
                st.session_state.processed_chunks = process_context_chunks(user_input)

        # Now generate a final answer
        combined_context = st.session_state.processed_chunks
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def process_final_answer():
                full_response = ""
                async for token in get_final_answer(combined_context, user_input):
                    full_response += token
                    response_placeholder.markdown(
                        f'<div class="assistant-message">{full_response}</div>', 
                        unsafe_allow_html=True
                    )
            loop.run_until_complete(process_final_answer())
        finally:
            loop.close()
            
        # Rerun the app to refresh the UI
        st.rerun()

        # Once the script re-runs from the top, you'll see the conversation updated
        # with the user's message and the final response from the assistant.

# Replace the async processing in main() with:
def process_chunk_sync(chunk_and_question):
    """
    Synchronous version of chunk processing for multiprocessing
    """
    chunk, question = chunk_and_question
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    
    # Build conversation history for context
    messages = [
        {
            "role": "system",
            "content": (
                "You are analyzing a chunk of text to copy the "
                "relevant context to the question provided by the user. "
                "If no relevant context is found, just output "
                "'no relevant answer' and no other explanation."
            )
        },
        {
            "role": "user",
            "content": (
                f"Based on this text:\n\n{chunk}\n\n"
                f"Find and copy the relevant context to answer this question: {question}"
            )
        }
    ]
    
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Local Script",
            },
            model=CONTEXT_GATHERER,
            messages=messages
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error processing chunk: {str(e)}"

async def get_final_answer(combined_context, question):
    """
    Use the combined context to get a final, direct answer from the model.
    """
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    # Build conversation history for context
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the provided <context> to answer "
                "the user's question. Maintain a natural conversational flow. If you don't have enough context, just say you need more context."
            )
        }
    ]
    
    # Add conversation history
    for msg in st.session_state.conversation_history[-4:]:  # Include last 3-4 exchanges for context
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current context and question
    messages.append({
        "role": "user",
        "content": (
            f"Based on this context:\n\n<context>{combined_context}</context>\n\n"
            f"Please answer this question: {question}"
        )
    })

    try:
        completion = await client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Local Script",
            },
            model=CHAT_MODEL,
            messages=messages,
            stream=True
        )
        
        collected_messages = []
        async for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                collected_messages.append(chunk.choices[0].delta.content)
                yield chunk.choices[0].delta.content
                
        # After completion, update conversation history
        full_response = "".join(collected_messages)
        st.session_state.conversation_history.append({
            "role": "assistant", 
            "content": full_response,
            "chunks": combined_context  # Store chunks with the response
        })
        
    except Exception as e:
        error_msg = f"Error getting final answer: {str(e)}"
        st.session_state.conversation_history.append({
            "role": "assistant", 
            "content": error_msg,
            "chunks": combined_context  # Store chunks even with error messages
        })
        yield error_msg

# Add new process_context_chunks function
def process_context_chunks(user_input):
    """
    Splits the context into overlapping chunks and processes them in parallel.
    """
    chunk_size = 32000
    overlap = 1000  # Characters to overlap between chunks
    text = st.session_state.context_text
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    chunk_question_pairs = [(chunk, user_input) for chunk in chunks]

    relevant_contexts = []
    total_chunks = len(chunk_question_pairs)
    progress_bar = st.progress(0)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_chunk_sync, pair): pair for pair in chunk_question_pairs}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if not result.startswith("Error") and result != "no relevant answer":
                relevant_contexts.append(result)
            progress_bar.progress((i + 1) / total_chunks)
    progress_bar.empty()
    return "\n".join(relevant_contexts).strip()

if __name__ == "__main__":
    main()
