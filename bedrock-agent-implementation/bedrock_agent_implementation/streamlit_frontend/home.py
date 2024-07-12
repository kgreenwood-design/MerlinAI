import streamlit as st
import streamlit_authenticator as stauth
import boto3
import os
import random
import string
from PIL import Image
import speech_recognition as sr
from boto3.dynamodb.conditions import Key
import time
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
from yaml.loader import SafeLoader
import json
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import spacy
import spacy
import subprocess
import sys

def download_spacy_model():
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    download_spacy_model()
    nlp = spacy.load("en_core_web_sm")

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

def load_and_process_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    return df

def plot_data(df, plot_type, x_axis, y_axis):
    if plot_type == "Line Plot":
        fig = px.line(df, x=x_axis, y=y_axis)
    elif plot_type == "Bar Plot":
        fig = px.bar(df, x=x_axis, y=y_axis)
    elif plot_type == "Scatter Plot":
        fig = px.scatter(df, x=x_axis, y=y_axis)
    elif plot_type == "Box Plot":
        fig = px.box(df, x=x_axis, y=y_axis)
    else:
        st.error("Unsupported plot type.")
        return None
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(args=["type", "scatter"], label="Scatter", method="restyle"),
                    dict(args=["type", "line"], label="Line", method="restyle"),
                    dict(args=["type", "bar"], label="Bar", method="restyle")
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    return fig

def perform_advanced_analysis(df):
    # Prepare data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=pca_result[:, :2], columns=['PC1', 'PC2'])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_result = kmeans.fit_predict(X_scaled)

    # Create visualizations
    pca_fig = px.scatter(pca_df, x='PC1', y='PC2', color=kmeans_result, 
                         title='PCA with K-means Clustering')
    
    explained_variance = px.bar(x=range(1, len(pca.explained_variance_ratio_)+1),
                                y=pca.explained_variance_ratio_,
                                title='Explained Variance Ratio')

    return pca_fig, explained_variance

def save_visualization(fig, filename):
    with open(filename, 'wb') as f:
        pickle.dump(fig, f)

def load_visualization(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def parse_data_query(query):
    doc = nlp(query)
    plot_type = None
    columns = []
    
    for token in doc:
        if token.text.lower() in ["line", "bar", "scatter", "box"]:
            plot_type = token.text.lower() + " plot"
        if token.dep_ == "dobj" and token.head.pos_ == "VERB":
            columns.append(token.text)
    
    return plot_type, columns

def save_user_data(username, data):
    if not os.path.exists('user_data'):
        os.makedirs('user_data')
    with open(f'user_data/{username}.json', 'w') as f:
        json.dump(data, f)

def load_user_data(username):
    try:
        with open(f'user_data/{username}.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

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

def display_image(image_path):
    try:
        image = Image.open(image_path)
        st.image(image, caption="IoT Device Visualization", use_column_width=True)
    except FileNotFoundError:
        st.error(f"Image file not found: {image_path}")
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

def session_generator():
    # Generate random characters and digits
    digits = ''.join(random.choice(string.digits) for _ in range(4))  # Generating 4 random digits
    chars = ''.join(random.choice(string.ascii_lowercase) for _ in range(3))  # Generating 3 random characters
    
    # Construct the pattern (1a23b-4c)
    pattern = f"{digits[0]}{chars[0]}{digits[1:3]}{chars[1]}-{digits[3]}{chars[2]}"
    print("Session ID: " + str(pattern))

    return pattern

def check_session_timeout():
    if 'last_activity' in st.session_state:
        last_activity = st.session_state.last_activity
        if datetime.now() - last_activity > timedelta(minutes=10):
            st.session_state.authentication_status = None
            st.session_state.username = None
            st.session_state.name = None
            st.session_state.session_id = None
            st.warning("Your session has expired due to inactivity. Please log in again.")
            st.experimental_rerun()
    st.session_state.last_activity = datetime.now()

def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None
import yaml
from PIL import Image
from yaml.loader import SafeLoader

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

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
DYNAMODB_TABLE_NAME = os.getenv('DYNAMODB_TABLE_NAME', 'default_conversation_table')
client = boto3.client('bedrock-agent-runtime')
dynamodb = boto3.resource('dynamodb')

if DYNAMODB_TABLE_NAME:
    conversation_table = dynamodb.Table(DYNAMODB_TABLE_NAME)
else:
    st.warning("DYNAMODB_TABLE_NAME environment variable is not set. Some features may not work correctly.")
    conversation_table = None

# Render the login widget
authenticator.login()

if st.session_state["authentication_status"]:
    with st.sidebar:
        # Load and display the logo image
        logo = Image.open("image.png")
        st.image(logo, width=200)  # Adjust width as needed
        
        st.write(f'Welcome *{st.session_state["name"]}*')
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

def save_conversation(session_id, role, content):
    if conversation_table:
        conversation_table.put_item(
            Item={
                'session_id': session_id,
                'timestamp': int(time.time() * 1000),
                'role': role,
                'content': content
            }
        )
    else:
        st.warning("Unable to save conversation: DynamoDB table not configured.")

def get_conversation_history(session_id):
    if conversation_table:
        response = conversation_table.query(
            KeyConditionExpression=Key('session_id').eq(session_id),
            ScanIndexForward=True
        )
        return response['Items']
    else:
        st.warning("Unable to retrieve conversation history: DynamoDB table not configured.")
        return []

def main():
    # Check for session timeout
    check_session_timeout()

    # Add custom CSS for layout
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
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
    .stSelectbox {
        margin-bottom: 1rem;
    }
    .stPlotlyChart {
        margin-top: 2rem;
    }
    .st-emotion-cache-16idsys p {
        font-size: 14px;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize the session ID
    if 'session_id' not in st.session_state:
        st.session_state.session_id = session_generator()

    # Load conversation history from DynamoDB
    conversation_history = get_conversation_history(st.session_state.session_id)

    if st.session_state["authentication_status"]:
        st.title("MerlinAI")

        # Load user-specific data
        user_data = load_user_data(st.session_state["username"])

        # Update last activity time
        st.session_state.last_activity = datetime.now()

        # Add tabs for different functionalities
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Chat", "Device Metrics", "Data Visualization", "Data Analysis", "Advanced Analysis", "Image Display"])
        
        with tab1:
            # Display conversation history
            for interaction in conversation_history:
                if interaction['role'] == 'user':
                    st.markdown(f'<div class="chat-message user"><img src="https://via.placeholder.com/40" class="avatar"><div class="message">{interaction["content"]}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant"><img src="https://via.placeholder.com/40" class="avatar"><div class="message">{interaction["content"]}</div></div>', unsafe_allow_html=True)

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
                        # Clear conversation history from DynamoDB
                        clear_conversation_history(st.session_state.session_id)
                        st.experimental_rerun()
                with col3:
                    if st.button("New Session"):
                        st.session_state.session_id = session_generator()
                        st.experimental_rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            device_id = st.text_input("Enter Device ID")
            if device_id:
                plot_device_metrics(device_id)
        
        with tab3:
            st.subheader("Data Visualization")
            uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
            if uploaded_file is not None:
                df = load_and_process_data(uploaded_file)
                if df is not None:
                    st.write(df.head())
                    plot_type = st.selectbox("Select plot type", ["Line Plot", "Bar Plot", "Scatter Plot", "Box Plot"])
                    x_axis = st.selectbox("Select X-axis", df.columns)
                    y_axis = st.selectbox("Select Y-axis", df.columns)
                    fig = plot_data(df, plot_type, x_axis, y_axis)
                    if fig:
                        st.plotly_chart(fig)
                        if st.button("Save Visualization"):
                            save_visualization(fig, f"visualizations/{st.session_state['username']}_{plot_type}.pkl")
                            st.success("Visualization saved!")

        with tab4:
            st.subheader("Data Analysis")
            if 'df' in locals():
                st.write("Data Summary:")
                st.write(df.describe())
                
                st.write("Correlation Matrix:")
                corr_matrix = df.corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True)
                st.plotly_chart(fig_corr)
                
                st.write("Data Distribution:")
                column = st.selectbox("Select a column for distribution analysis", df.columns)
                fig_dist = px.histogram(df, x=column, marginal="box")
                st.plotly_chart(fig_dist)

        with tab5:
            st.subheader("Advanced Analysis")
            if 'df' in locals():
                if st.button("Perform Advanced Analysis"):
                    pca_fig, explained_variance = perform_advanced_analysis(df)
                    st.plotly_chart(pca_fig)
                    st.plotly_chart(explained_variance)

        with tab6:
            image_path = st.text_input("Enter the path to your image:")
            if image_path:
                display_image(image_path)

    else:
        st.warning("Please log in to access the application.")

def process_user_input(user_prompt):
    try:
        # Save user message to DynamoDB
        save_conversation(st.session_state.session_id, 'user', user_prompt)
        
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
        
        # Save assistant's response to DynamoDB
        save_conversation(st.session_state.session_id, 'assistant', answer)
        
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error occurred when calling MultiRouteChain. Please review application logs for more information.")
        print(f"ERROR: Exception when calling MultiRouteChain: {e}")
        # Save error message to DynamoDB
        save_conversation(st.session_state.session_id, 'assistant', f"Error occurred: {e}")
        st.experimental_rerun()

def extract_device_id(text):
    # Implement logic to extract device ID from the text
    # This is a placeholder implementation
    import re
    match = re.search(r'device (\d+)', text, re.IGNORECASE)
    return match.group(1) if match else None

if __name__ == '__main__':
    main()
def clear_conversation_history(session_id):
    # Query all items for the session
    response = conversation_table.query(
        KeyConditionExpression=Key('session_id').eq(session_id)
    )
    
    # Delete each item
    with conversation_table.batch_writer() as batch:
        for item in response['Items']:
            batch.delete_item(
                Key={
                    'session_id': item['session_id'],
                    'timestamp': item['timestamp']
                }
            )
