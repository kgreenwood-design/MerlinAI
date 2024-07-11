import streamlit as st
import streamlit_authenticator as stauth
import boto3
import os
import random
import string
from PIL import Image
import speech_recognition as sr
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_device_metrics(device_id):
    # This is a placeholder. In reality, you'd fetch this data from your database
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
        'temperature': np.random.randn(100) * 10 + 25,
        'pressure': np.random.randn(100) * 5 + 100,
        'oil_level': np.random.randn(100) * 2 + 50
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['temperature'], name='Temperature'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['pressure'], name='Pressure'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['oil_level'], name='Oil Level'))
    
    fig.update_layout(title=f'Metrics for Device {device_id}', xaxis_title='Time', yaxis_title='Value')
    st.plotly_chart(fig)

def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Say something!")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        st.write(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        st.write("Could not understand audio")
    except sr.RequestError as e:
        st.write(f"Could not request results; {e}")

# Add this after the imports
def display_image():
    image = Image.open("path_to_your_image.jpg")
    st.image(image, caption="IoT Device Visualization", use_column_width=True)

def session_generator():
    # Generate random characters and digits
    digits = ''.join(random.choice(string.digits) for _ in range(4))  # Generating 4 random digits
    chars = ''.join(random.choice(string.ascii_lowercase) for _ in range(3))  # Generating 3 random characters
    
    # Construct the pattern (1a23b-4c)
    pattern = f"{digits[0]}{chars[0]}{digits[1:3]}{chars[1]}-{digits[3]}{chars[2]}"
    print("Session ID: " + str(pattern))

    return pattern
import yaml
from PIL import Image
from yaml.loader import SafeLoader

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .login-container {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .logo-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Load the logo image
logo = Image.open("image.png")

# Create the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Create a container for the login form and logo
login_container = st.container()

with login_container:
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image(logo, width=300)  # Increased width to make the image bigger
    st.markdown('</div>', unsafe_allow_html=True)

with login_container:
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
BEDROCK_AGENT_ID = os.getenv('BEDROCK_AGENT_ID')
BEDROCK_AGENT_ALIAS = os.getenv('BEDROCK_AGENT_ALIAS')
client = boto3.client('bedrock-agent-runtime')

# Render the login widget
authenticator.login()

if st.session_state["authentication_status"]:
    st.markdown('</div>', unsafe_allow_html=True)  # Close login-container
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.markdown("<h1 style='text-align: center; color: #4A90E2; font-family: sans-serif;'>MerlinAI</h1>", unsafe_allow_html=True)

    with st.sidebar:
        authenticator.logout('Logout', 'main')

        with st.expander("Account Settings"):
            # Password reset widget
            try:
                if authenticator.reset_password(st.session_state["username"], fields={'Form name':'Reset password', 'Current password':'Current password', 'New password':'New password', 'Repeat password':'Repeat password', 'Reset':'Reset'}):
                    st.success('Password modified successfully')
            except Exception as e:
                st.error(e)

            # Update user details widget
            try:
                if authenticator.update_user_details(st.session_state["username"], fields={'Form name':'Update user details', 'Field':'Field', 'Name':'Name', 'Email':'Email', 'New value':'New value', 'Update':'Update'}):
                    st.success('Entries updated successfully')
            except Exception as e:
                st.error(e)

        with st.expander("Session Options"):
            st.write("Session ID: ", st.session_state.session_id)
            if st.button("Generate New Session ID", key="generate_session_id_sidebar"):
                st.session_state.session_id = session_generator()
                st.experimental_rerun()

        with st.expander("Conversation Options"):
            if st.button("Clear Conversation", key="clear_conversation_sidebar"):
                st.session_state.conversation = []
                st.experimental_rerun()

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')

if not st.session_state["authentication_status"]:
    if 'show_forgot_password' not in st.session_state:
        st.session_state.show_forgot_password = False
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False

    col1, col2 = st.columns([1, 1])

    if st.session_state.show_forgot_password:
        with col1:
            # Forgotten password widget
            try:
                username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password(fields={'Form name':'Forgot password', 'Username':'Username', 'Submit':'Submit'})
                if username_forgot_pw:
                    st.success('New password sent securely')
                    st.session_state.show_forgot_password = False
                    # Random password to be transferred to user securely
                elif username_forgot_pw == False:
                    st.error('Username not found')
            except Exception as e:
                st.error(e)

    if st.session_state.show_register:
        with col2:
            # New user registration widget
            try:
                email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(fields={'Form name':'Register user', 'Email':'Email', 'Username':'Username', 'Password':'Password', 'Repeat password':'Repeat password', 'Register':'Register'})
                st.success('User registered successfully')
                st.session_state.show_register = False
            except Exception as e:
                st.error(e)

    with col1:
        if st.button('Forgot Password?', key="forgot_password_button"):
            st.session_state.show_forgot_password = not st.session_state.show_forgot_password
            st.session_state.show_register = False
    with col2:
        if st.button('Register', key="register_button"):
            st.session_state.show_register = not st.session_state.show_register
            st.session_state.show_forgot_password = False

st.markdown('</div>', unsafe_allow_html=True)  # Close login-container


def format_retrieved_references(references):
    # Extracting the text and link from the references
    for reference in references:
        content_text = reference.get("content", {}).get("text", "")
        s3_uri = reference.get("location", {}).get("s3Location", {}).get("uri", "")

        # Formatting the output
        formatted_output = "Reference Information:\n"
        formatted_output += f"Content: {content_text}\n"
        formatted_output += f"S3 URI: {s3_uri}\n"

        return formatted_output


def process_stream(stream):
    try:
        # print("Processing stream...")
        trace = stream.get("trace", {}).get("trace", {}).get("orchestrationTrace", {})

        if trace:
            # print("This is a trace")
            knowledgeBaseInput = trace.get("invocationInput", {}).get(
                "knowledgeBaseLookupInput", {}
            )
            if knowledgeBaseInput:
                print(
                    f'Looking up in knowledgebase: {knowledgeBaseInput.get("text", "")}'
                )
            knowledgeBaseOutput = trace.get("observation", {}).get(
                "knowledgeBaseLookupOutput", {}
            )
            if knowledgeBaseOutput:
                retrieved_references = knowledgeBaseOutput.get(
                    "retrievedReferences", {}
                )
                if retrieved_references:
                    print("Formatted References:")
                    return format_retrieved_references(retrieved_references)

        # Handle 'chunk' data
        if "chunk" in stream:
            print("This is the final answer:")
            text = stream["chunk"]["bytes"].decode("utf-8")
            return text

    except Exception as e:
        print(f"Error processing stream: {e}")
        print(stream)

def session_generator():
    # Generate random characters and digits
    digits = ''.join(random.choice(string.digits) for _ in range(4))  # Generating 4 random digits
    chars = ''.join(random.choice(string.ascii_lowercase) for _ in range(3))  # Generating 3 random characters
    
    # Construct the pattern (1a23b-4c)
    pattern = f"{digits[0]}{chars[0]}{digits[1:3]}{chars[1]}-{digits[3]}{chars[2]}"
    print("Session ID: " + str(pattern))

    return pattern

def main():
    # Add custom CSS for layout
    st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #f0f0f0;
    }
    .chat-message.assistant {
        background-color: #e6f3ff;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: white;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Add file uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)

    # Add tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Chat", "Device Metrics", "Voice Input"])
    
    with tab1:
        # Your existing chat functionality goes here
        pass
    
    with tab2:
        device_id = st.text_input("Enter Device ID")
        if device_id:
            plot_device_metrics(device_id)
    
    with tab3:
        if st.button("Start Voice Input"):
            user_input = voice_input()
            if user_input:
                process_user_input(user_input)

    # Display image
    display_image()

    # Initialize the conversation state and session ID
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = session_generator()

    if st.session_state["authentication_status"]:
        st.title("MerlinAI")

        # Display conversation
        for interaction in st.session_state.conversation:
            if 'user' in interaction:
                st.markdown(f'<div class="chat-message user"><img src="https://via.placeholder.com/40" class="avatar"><div class="message">{interaction["user"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant"><img src="https://via.placeholder.com/40" class="avatar"><div class="message">{interaction["assistant"]}</div></div>', unsafe_allow_html=True)

        # Input container
        with st.container():
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            user_prompt = st.text_input("Message:", key="user_input")
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                if st.button("Submit"):
                    if user_prompt:
                        process_user_input(user_prompt)
            with col2:
                if st.button("Clear Conversation"):
                    st.session_state.conversation = []
                    st.experimental_rerun()
            with col3:
                if st.button("New Session"):
                    st.session_state.session_id = session_generator()
                    st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)

def process_user_input(user_prompt):
    try:
        st.session_state.conversation.append({'user': user_prompt})
        response = client.invoke_agent(
            agentId=BEDROCK_AGENT_ID,
            agentAliasId=BEDROCK_AGENT_ALIAS,
            sessionId=st.session_state.session_id,
            endSession=False,
            inputText=user_prompt
        )
        results = response.get("completion", [])
        answer = ""
        for stream in results:
            answer += process_stream(stream)
        
        # Check if the answer contains any actionable commands
        if "display_metrics" in answer.lower():
            device_id = extract_device_id(answer)  # You need to implement this function
            plot_device_metrics(device_id)
        elif "upload_file" in answer.lower():
            st.info("Please use the file uploader in the sidebar to upload a CSV file.")
        
        st.session_state.conversation.append({'assistant': answer})
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error occurred when calling MultiRouteChain. Please review application logs for more information.")
        print(f"ERROR: Exception when calling MultiRouteChain: {e}")
        st.session_state.conversation.append({'assistant': f"Error occurred: {e}"})
        st.experimental_rerun()

def extract_device_id(text):
    # Implement logic to extract device ID from the text
    # This is a placeholder implementation
    import re
    match = re.search(r'device (\d+)', text, re.IGNORECASE)
    return match.group(1) if match else None

if __name__ == '__main__':
    main()
