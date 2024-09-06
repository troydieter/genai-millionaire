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
import wave
import time
import requests

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

# instantiating the S3 client
s3 = boto3.client('s3')

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

def transcribe_audio(audio_data):
    # Create a temporary WAV file
    with io.BytesIO() as wav_file:
        with wave.open(wav_file, 'wb') as wav:
            wav.setnchannels(1)  # mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(16000)  # 16kHz
            wav.writeframes(audio_data)
        wav_file.seek(0)
        
        # Upload the WAV file to S3
        bucket_name = os.getenv('S3_BUCKET_NAME')  # Get S3 bucket name from environment variable
        file_name = f'temp_audio_{int(time.time())}.wav'
        s3.upload_fileobj(wav_file, bucket_name, file_name)

    # Start transcription job
    job_name = f"transcription_job_{int(time.time())}"
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': f"s3://{bucket_name}/{file_name}"},
        MediaFormat='wav',
        LanguageCode='en-US'
    )

    # Wait for the transcription job to complete
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(5)

    # Clean up the S3 file
    s3.delete_object(Bucket=bucket_name, Key=file_name)

    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        result = requests.get(status['TranscriptionJob']['Transcript']['TranscriptFileUri']).json()
        transcript = result['results']['transcripts'][0]['transcript']
        return transcript
    else:
        return "Transcription failed"

# Sidebar controls - Select your lifeline!
with st.sidebar:
    # Start the "Call a Friend" transcription job
    def processing():
        with st.spinner(':telephone_receiver: Calling a friend...'):
            global transcript
            audio_data = b''.join(list(audio_buffer.queue))
            transcript = transcribe_audio(audio_data)
            if not transcript or transcript == "Transcription failed":
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