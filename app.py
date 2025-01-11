import os
import asyncio
import streamlit as st
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
# Note: You can use https://gitingest.com/ to securely copy and manage your tokens
load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Model constants
CONTEXT_GATHERER = "deepseek/deepseek-chat" # higher throughput model is preferred here
CHAT_MODEL = "deepseek/deepseek-chat" # better model is preferred here

###############################################################################
# Async function that processes one chunk
###############################################################################
async def process_chunk_async(chunk, question):
    """
    Send one chunk to the OpenRouter API asynchronously and return the response.
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
                "You are analyzing a chunk of text to copy the "
                "relevant context to the question provided by the user. "
                "If no relevant context is found, just output "
                "'no relevant answer' and no other explanation."
            )
        }
    ]
    
    # Add conversation history
    for msg in st.session_state.conversation_history[-3:]:  # Include last 3 exchanges for context
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current question
    messages.append({
        "role": "user",
        "content": (
            f"Based on this text:\n\n{chunk}\n\n"
            f"Gather the relevant context to answer the following question, "
            f"taking into account our previous conversation: {question}"
        )
    })
    
    try:
        completion = await client.chat.completions.create(
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

# Async function that processes all chunks concurrently using gather
async def process_all_chunks_async(chunks, question):
    tasks = [process_chunk_async(chunk, question) for chunk in chunks]
    return await asyncio.gather(*tasks)

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
                "You are a helpful assistant. Use the provided context to answer "
                "the user's question. Maintain a natural conversational flow."
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
            f"Based on this context:\n\n{combined_context}\n\n"
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
        st.session_state.conversation_history.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        error_msg = f"Error getting final answer: {str(e)}"
        st.session_state.conversation_history.append({"role": "assistant", "content": error_msg})
        yield error_msg

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
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'context_text' not in st.session_state:
        st.session_state.context_text = ""
    if 'processed_chunks' not in st.session_state:
        st.session_state.processed_chunks = None
    
    # Sidebar for context input
    with st.sidebar:
        st.title("Context Settings")
        st.info("💡 Use [gitingest.com](https://gitingest.com/) for secure token management.")
        new_context = st.text_area("Enter your context text here:", height=300)
        if st.button("Set Context"):
            st.session_state.context_text = new_context
            st.session_state.processed_chunks = None  # Reset processed chunks when context changes
            st.success("Context updated successfully!")
    
    st.title("fastAsk")
    
    # 1) Display the conversation so far
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

    # 2) Add a form to accept new user input, but only *after* everything above is shown
    #    This means the form will appear at the bottom, under the last assistant message.
    user_input = None
    with st.form(key='message_form'):
        user_input = st.text_input("Enter your message here:")
        submit_button = st.form_submit_button("Send")

    # 3) If the user provided a new message, handle it
    if submit_button and user_input:
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

        # Process chunks if we haven't done so yet
        if st.session_state.processed_chunks is None:
            chunk_size = 16000
            chunks = [
                st.session_state.context_text[i : i + chunk_size] 
                for i in range(0, len(st.session_state.context_text), chunk_size)
            ]
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def run_async():
                    with st.spinner('Processing context...'):
                        return await process_all_chunks_async(chunks, user_input)
                
                responses = loop.run_until_complete(run_async())
                
                # Filter out errors and "no relevant answer"
                relevant_contexts = [
                    resp for resp in responses 
                    if not resp.startswith("Error") and resp != "no relevant answer"
                ]
                
                st.session_state.processed_chunks = "\n".join(relevant_contexts).strip()
            finally:
                loop.close()

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

        # Once the script re-runs from the top, you’ll see the conversation updated
        # with the user's message and the final response from the assistant.

if __name__ == "__main__":
    main()
