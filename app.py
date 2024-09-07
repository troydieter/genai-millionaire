import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import os
from prompt_finder_and_invoke_llm import prompt_finder
from chat_history_prompt_generator import chat_history
from live_transcription import main
from dotenv import load_dotenv
import boto3
import botocore.config
import threading

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

# Title displayed on the streamlit web app
st.title(f""":rainbow[Bedrock Speech-to-Text Chat]""")

# WebRTC audio stream processing class
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.transcript = ""
        self.running = False

    def recv(self, frames, dtype):
        # Convert audio frames to numpy array
        audio_data = np.frombuffer(frames, dtype=np.float32)

        # Here you can process the audio (like sending it to AWS Transcribe)
        # For now, just simulate transcription
        if not self.running:
            self.running = True
            threading.Thread(target=self.transcribe_audio).start()

        return frames

    def transcribe_audio(self):
        # Call the transcription function (can integrate live_transcription.main() here)
        with st.spinner(':ear: Bedrock is listening...'):
            global transcript
            transcript = main("en-US")  # Call the transcription function from your script
        st.session_state.transcript = transcript
        self.running = False

# configuring values for session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.transcript = ""
    st.session_state.run = False
    open("chat_history.txt", "w").close()
# writing the message that is stored in session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Placeholder for dynamic responses
response_placeholder = st.empty()

# Sidebar UI for controlling WebRTC microphone input and transcription
with st.sidebar:
    # Button to start WebRTC streaming and transcription
    def start_webrtc():
        st.session_state.run = True

    # Button to reset the conversation and states
    def clear():
        global response_placeholder
        response_placeholder = st.empty()
        st.session_state.result = None
        st.session_state.transcript = ""
        st.session_state.run = False

    st.button('Start Conversation', type="primary", on_click=start_webrtc, disabled=st.session_state.run)
    st.button('Clear Conversation', on_click=clear)

# Only start WebRTC streaming if the user clicks "Start Conversation"
if st.session_state.run:
    webrtc_ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDONLY,  # Use WebRtcMode enum instead of string
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

# Evaluate transcript and handle responses
if st.session_state.transcript:
    with st.chat_message("user"):
        st.markdown(st.session_state.transcript)
        st.balloons()

    st.session_state.messages.append({"role": "user", "content": st.session_state.transcript})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Determining the best possible answer!"):
            answer = prompt_finder(st.session_state.transcript)
            message_placeholder.markdown(f"{answer}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Convert response to audio via Polly and play
    response = polly.synthesize_speech(Text=answer, OutputFormat='mp3', VoiceId='Danielle', Engine="neural")
    response_audio = response['AudioStream'].read()
    response_placeholder.audio(response_audio, format='audio/mp3', start_time=0, autoplay=True)

    # Update chat history for future prompts
    chat_history(st.session_state)
