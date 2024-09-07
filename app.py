import streamlit as st
import os
from prompt_finder_and_invoke_llm import prompt_finder
from chat_history_prompt_generator import chat_history
from dotenv import load_dotenv
import boto3
import botocore.config
import asyncio
import websockets
import json
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder
import uuid

load_dotenv()

boto3.setup_default_session(profile_name=os.getenv('profile_name'))
config = botocore.config.Config(connect_timeout=120, read_timeout=120)
polly = boto3.client('polly', region_name='us-east-1')
transcribe = boto3.client('transcribe', region_name='us-east-1')

st.title(f""":rainbow[Bedrock Speech-to-Text Chat]""")

if "messages" not in st.session_state:
    st.session_state.messages = []
    open("chat_history.txt", "w").close()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

transcript = ""
response_placeholder = st.empty()

def play_audio(audio_data):
    st.audio(audio_data, format='audio/mp3', start_time=0, autoplay=True)

async def process_audio(audio_file_path):
    job_name = f"Transcription-{uuid.uuid4()}"
    with open(audio_file_path, "rb") as audio_file:
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': f"file://{audio_file_path}"},
            MediaFormat='wav',
            LanguageCode='en-US'
        )

    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        await asyncio.sleep(5)

    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        result = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        transcript_uri = result['TranscriptionJob']['Transcript']['TranscriptFileUri']
        transcript_response = requests.get(transcript_uri)
        transcript_data = transcript_response.json()
        return transcript_data['results']['transcripts'][0]['transcript']
    else:
        return "Transcription failed"

async def webrtc_audio():
    pc = RTCPeerConnection()
    recorder = MediaRecorder("audio.wav")

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            recorder.addTrack(track)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # In a real-world scenario, you'd exchange the offer and answer with the client
    # For simplicity, we're creating a dummy answer
    answer = RTCSessionDescription(sdp="dummy_sdp", type="answer")
    await pc.setRemoteDescription(answer)

    await recorder.start()
    
    # Record audio for 10 seconds
    await asyncio.sleep(10)
    
    await recorder.stop()
    
    transcript = await process_audio("audio.wav")
    return transcript

with st.sidebar:
    if st.button('Start Recording'):
        with st.spinner('Recording and transcribing...'):
            transcript = asyncio.run(webrtc_audio())
        st.success('Recording and transcription complete!')

if transcript:
    with st.chat_message("user"):
        st.markdown(transcript)
        st.balloons()
    
    st.session_state.messages.append({"role": "user", "content": transcript})
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Determining the best possible answer!"):
            answer = prompt_finder(transcript)
            message_placeholder.markdown(f"{answer}")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    response = polly.synthesize_speech(Text=answer, OutputFormat='mp3', VoiceId='Danielle', Engine="neural")
    response_audio = response['AudioStream'].read()
    response_placeholder = st.empty()
    response_placeholder.audio(response_audio, format='audio/mp3', start_time=0, autoplay=True)
    
    chat_history(st.session_state)