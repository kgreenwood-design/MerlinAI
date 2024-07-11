import streamlit as st
import streamlit_authenticator as stauth
import boto3
import os
import random
import string

import yaml
from PIL import Image
from yaml.loader import SafeLoader

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
BEDROCK_AGENT_ID = os.getenv('BEDROCK_AGENT_ID')
BEDROCK_AGENT_ALIAS = os.getenv('BEDROCK_AGENT_ALIAS')
client = boto3.client('bedrock-agent-runtime')

# Render the login widget
authenticator.login()

if st.session_state["authentication_status"]:
    with st.sidebar:
        authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title('Conversational AI - Plant Technician')

    with st.sidebar:
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

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')

    if 'show_forgot_password' not in st.session_state:
        st.session_state.show_forgot_password = False
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False

    col1, col2 = st.columns(2)

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
        if st.button('Forgot Password?'):
            st.session_state.show_forgot_password = not st.session_state.show_forgot_password
            st.session_state.show_register = False
    with col2:
        if st.button('Register'):
            st.session_state.show_register = not st.session_state.show_register
            st.session_state.show_forgot_password = False

elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

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

    # Initialize the conversation state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    # Initialize the agent session id
    if 'session_id' not in st.session_state:
        st.session_state.session_id = session_generator()

    if st.session_state["authentication_status"]:
        # Display the logo image within the message box
        st.image(logo, width=100)

        # Taking user input        
        user_prompt = st.text_area("Message:", height=150)

        if user_prompt and st.button("Submit"):
            try:
                # Add the user's prompt to the conversation state
                st.session_state.conversation.append({'user': user_prompt})

                # Format and add the answer to the conversation state
                response = client.invoke_agent(
                    agentId=BEDROCK_AGENT_ID,
                    agentAliasId=BEDROCK_AGENT_ALIAS,
                    sessionId=st.session_state.session_id,
                    endSession=False,
                    inputText=user_prompt
                )
                results = response.get("completion")
                answer = ""
                for stream in results:
                    answer += process_stream(stream)
                st.session_state.conversation.append(
                    {'assistant': answer})

                # Clear the text input box after submission
                st.experimental_rerun()

            except Exception as e:
                # Clear the text input box if an error occurs
                st.experimental_rerun()
                # Display an error message if an exception occurs
                st.error(f"Error occurred when calling MultiRouteChain. Please review application logs for more information.")
                print(f"ERROR: Exception when calling MultiRouteChain: {e}")
                formatted_answer = f"Error occurred: {e}"
                st.session_state.conversation.append(
                    {'assistant': formatted_answer})

        # Display the conversation
        for interaction in st.session_state.conversation:
            with st.container():
                if 'user' in interaction:
                    # Apply a custom color to the "User" alias using inline CSS
                    st.markdown(f'<span style="color: #4A90E2; font-weight: bold;">User:</span> {interaction["user"]}', unsafe_allow_html=True)
                else:
                    # Apply a different custom color to the "Assistant" alias using inline CSS
                    st.markdown(f'<span style="color: #50E3C2; font-weight: bold;">Assistant:</span> {interaction["assistant"]}', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
