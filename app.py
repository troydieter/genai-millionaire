import streamlit as st
import os
from prompt_finder_and_invoke_llm import prompt_finder
from chat_history_prompt_generator import chat_history
from dotenv import load_dotenv
import boto3
import botocore.config
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import queue
import threading
import io
import asyncio
import aiohttp
import json
import base64

# loading in environment variables
load_dotenv()

# configuring our CLI profile name
boto3.setup_default_session(profile_name=os.getenv('profile_name'))
# increasing the timeout period when invoking bedrock
config = botocore.config.Config(connect_timeout=120, read_timeout=120)
# instantiating the Polly client
polly = boto3.client('polly', region_name='us-east-1')
# instantiating the Transcribe client
transcribe = boto3.client('transcribe', region_name='us-east-1')

# Title displayed on the Streamlit web app
st.title(f""":money_with_wings: **Who Wants to Be an AI Millionaire?** :moneybag:""")

# configuring values for session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    open("chat_history.txt", "w").close()
# displaying chat messages stored in session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# creating empty transcript string for streamed input to be added to
transcript = ""
response_placeholder = st.empty()

# Function to synthesize speech using Polly
def synthesize_speech(text):
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Joanna',  # You can change the voice as needed
        Engine='neural'
    )
    
    if "AudioStream" in response:
        return response["AudioStream"].read()
    else:
        return None

# WebRTC configuration
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Global variable to store audio data
audio_buffer = queue.Queue()

def audio_frame_callback(frame):
    sound = frame.to_ndarray()
    audio_buffer.put(sound)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": False, "audio": True},
    audio_frame_callback=audio_frame_callback,
    video_frame_callback=video_frame_callback,
    async_processing=True,
)

async def transcribe_audio_stream(audio_data):
    # Convert audio data to base64
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

    # Prepare the request payload
    payload = {
        "AudioStream": {
            "AudioEvent": {
                "AudioChunk": audio_base64
            }
        }
    }

    # Set up the Transcribe streaming endpoint
    endpoint = f"https://transcribe-streaming.{os.getenv('AWS_REGION')}.amazonaws.com/stream-transcription-websocket"

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get('Transcript', {}).get('Results', [{}])[0].get('Alternatives', [{}])[0].get('Transcript', '')
            else:
                return ''

# Sidebar controls - Select your lifeline!
with st.sidebar:
    # Start the "Call a Friend" transcription job
    def processing():
        with st.spinner(':telephone_receiver: Calling a friend...'):
            global transcript
            audio_data = b''.join(audio_buffer.queue)
            
            # Use asyncio to run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            transcript = loop.run_until_complete(transcribe_audio_stream(audio_data))
            loop.close()

            if not transcript:
                transcript = "I'm sorry, I couldn't understand the question. Could you please repeat it?"

        return "Transcription ended!"
    
    # Check if the lifeline is active
    if 'run' not in st.session_state:
        st.session_state.run = False
        st.session_state.result = None

    # Activate the lifeline
    def run():
        st.session_state.run = True
    
    # Reset the game
    def clear():
        global response_placeholder
        response_placeholder = st.empty()
        st.session_state.result = None
    
    # Instructions for asking questions (using lifelines)
    upper = st.container()
    upper.write(':studio_microphone: Click to ask a question! After 3 seconds of silence, your AI friend will respond.')
    st.button('Call a Friend', type="primary", on_click=run, disabled=st.session_state.run)
    result_area = st.empty()

    # Start transcription when the button is clicked
    if st.session_state.run:
        result_area.empty()
        st.session_state.result = processing()
        st.session_state.run = False

    # Show a reset button when transcription ends
    if st.session_state.result == "Transcription ended!":
        result_container = result_area.container()
        result_container.write(st.session_state.result)
        result_container.button('Ask a New Question', on_click=clear)

# If transcript is available, display it as a question
if transcript:
    with st.chat_message("user"):
        st.markdown(f":question: **Your Question:** {transcript}")
        st.balloons()
    st.session_state.messages.append({"role": "user", "content": transcript})

    # Answer from the AI (the Millionaire Expert)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            answer = prompt_finder(transcript)
            message_placeholder.markdown(f":moneybag: **AI Expert's Answer:** {answer}")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Synthesize speech for the answer using Polly
    audio_data = synthesize_speech(answer)
    
    if audio_data:
        # Create an in-memory file-like object
        audio_file = io.BytesIO(audio_data)
        
        # Display audio player in Streamlit
        st.audio(audio_file, format='audio/mp3', start_time=0)
    else:
        st.error("Failed to synthesize speech.")

    # Append chat history for future references
    chat_history(st.session_state)