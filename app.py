import streamlit as st
import os
from prompt_finder_and_invoke_llm import prompt_finder
from chat_history_prompt_generator import chat_history
from live_transcription import main
from dotenv import load_dotenv
import boto3
import botocore.config
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
import pydub

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

# Function to play audio from Polly response
def play_audio(audio_data):
    st.audio(audio_data, format='audio/mp3', start_time=0, autoplay=True)

# Audio recording function
def audio_recorder_callback(frame):
    return frame

# WebRTC configuration
webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"audio": True, "video": False},
)

# Sidebar controls - Select your lifeline!
with st.sidebar:
    # Start the "Call a Friend" transcription job
    def processing():
        with st.spinner(':telephone_receiver: Calling a friend...'):
            global transcript
            transcript = main("en-US")
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

# If audio data is available from WebRTC, process it
if webrtc_ctx.audio_receiver:
    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
    if audio_frames:
        sound = pydub.AudioSegment.empty()
        for audio_frame in audio_frames:
            sound += pydub.AudioSegment(
                data=audio_frame.to_ndarray().tobytes(),
                sample_width=audio_frame.format.bytes,
                frame_rate=audio_frame.sample_rate,
                channels=len(audio_frame.layout.channels),
            )
        # Save the audio data
        sound.export("recorded_audio.wav", format="wav")
        st.audio("recorded_audio.wav")

        # Process the audio (e.g., transcribe it)
        with open("recorded_audio.wav", "rb") as audio_file:
            transcript = transcribe.transcribe_audio(audio_file)

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

    # Polly speaks the answer
    response = polly.synthesize_speech(Text=answer, OutputFormat='mp3', VoiceId='Danielle', Engine="neural")
    response_audio = response['AudioStream'].read()
    response_placeholder.audio(response_audio, format='audio/mp3', start_time=0, autoplay=True)

    # Append chat history for future references
    chat_history(st.session_state)