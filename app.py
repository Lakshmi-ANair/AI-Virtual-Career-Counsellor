import streamlit as st
import requests
import uuid # To generate unique session IDs

# --- Configuration ---
RASA_API_URL = "http://localhost:5005/webhooks/rest/webhook"
BOT_AVATAR = "🤖"
USER_AVATAR = "👤"

# --- Streamlit Page Setup ---
st.set_page_config(page_title="AI Career Counsellor", layout="centered")
st.title("AI Virtual Career Counsellor 💬")
st.caption("Your friendly AI assistant to help you explore career paths based on your interests!")

# --- Session State Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()) # Unique ID for Rasa session

if "messages" not in st.session_state:
    st.session_state.messages = [] # To store chat history {role: "user/assistant", content: "message"}

if "initial_greeting_sent" not in st.session_state:
    st.session_state.initial_greeting_sent = False


# --- Helper Functions ---
def send_message_to_rasa(message_text, sender_id):
    """Sends a message to the Rasa server and gets the response."""
    payload = {
        "sender": sender_id,
        "message": message_text
    }
    try:
        response = requests.post(RASA_API_URL, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Rasa server: {e}")
        return None

# --- Initial Greeting from Bot ---
# if not st.session_state.initial_greeting_sent:
    # You can send a dummy message like "start_conversation" or use a Rasa rule for initial greeting
    # For simplicity, let's assume Rasa greets on first actual user message, or we can hardcode one
  #  initial_bot_greeting = "Hello! I'm your AI Career Counsellor. Tell me about your interests or what you enjoy doing."
   # st.session_state.messages.append({"role": "assistant", "content": initial_bot_greeting, "avatar": BOT_AVATAR})
    #st.session_state.initial_greeting_sent = True


# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])


# --- User Input Handling ---
if prompt := st.chat_input("What are your interests? (e.g., coding, art, finance)"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": USER_AVATAR})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # Get bot response
    with st.spinner("Thinking..."):
        rasa_responses = send_message_to_rasa(prompt, st.session_state.session_id)

    if rasa_responses:
        for response in rasa_responses:
            bot_message_text = response.get("text", "Sorry, I had trouble understanding that.")
            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": bot_message_text, "avatar": BOT_AVATAR})
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                st.markdown(bot_message_text)
    else:
        # Handle case where Rasa server is down or returns no response
        error_message = "I'm having trouble connecting right now. Please try again later."
        st.session_state.messages.append({"role": "assistant", "content": error_message, "avatar": BOT_AVATAR})
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(error_message)

# --- Optional: Clear chat button ---
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4()) # Reset session ID
    st.session_state.initial_greeting_sent = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**About:** This chatbot uses Rasa for NLU and dialogue, and NLTK for text processing. It recommends careers based on your stated interests.")