import streamlit as st
import streamlit_authenticator as stauth
import boto3
import os
import yaml
from yaml.loader import SafeLoader

# Load configuration
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Create the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

BEDROCK_AGENT_ID = os.getenv('BEDROCK_AGENT_ID')
BEDROCK_AGENT_ALIAS = os.getenv('BEDROCK_AGENT_ALIAS')
client = boto3.client('bedrock-agent-runtime')

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .chat-container {
        height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 10px;
        margin-bottom: 20px;
    }
    .input-container {
        display: flex;
        gap: 10px;
    }
    .stTextInput > div > div > input {
        height: 50px;
    }
</style>
""", unsafe_allow_html=True)

# Authentication
authenticator.login()

if st.session_state["authentication_status"]:
    st.title("MerlinAI")

    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id =  ''.join(os.urandom(10).hex())

    # Chat display
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.conversation:
            st.write(f"{'You' if 'user' in message else 'Assistant'}: {list(message.values())[0]}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Input area
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            user_input = st.text_input("Enter your message:", key="user_input")
        with col2:
            if st.button("Send"):
                if user_input:
                    st.session_state.conversation.append({'user': user_input})
                    try:
                        response = client.invoke_agent(
                            agentId=BEDROCK_AGENT_ID,
                            agentAliasId=BEDROCK_AGENT_ALIAS,
                            sessionId=st.session_state.session_id,
                            endSession=False,
                            inputText=user_input
                        )
                        answer = response.get("completion", [])
                        st.session_state.conversation.append({'assistant': answer})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                    st.experimental_rerun()
        with col3:
            if st.button("Clear Chat"):
                st.session_state.conversation = []
                st.experimental_rerun()

    # Logout button in sidebar
    with st.sidebar:
        authenticator.logout('Logout', 'main')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
