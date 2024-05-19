import openai
import speech_recognition as sr
import ibm_watson
from ibm_watson import TextToSpeechV1
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import pyaudio
import wave

# OpenAI GPT-3 API Key
openai.api_key = 'sk-proj-A3dAVJUJU5ZNGFmqtDdYT3BlbkFJVROoF5TMYfcnJoKdyKoy'

# IBM Watson Text to Speech
tts_authenticator = IAMAuthenticator('KUOJmw4305bJMeQ0ZpOEPYyn-8gakEDD19mOsRHJ2K6Z')
text_to_speech = TextToSpeechV1(authenticator=tts_authenticator)
text_to_speech.set_service_url('https://api.au-syd.text-to-speech.watson.cloud.ibm.com/instances/83e04dcf-07c6-4bd5-bd0f-c15f890e169f')

# IBM Watson Speech to Text
stt_authenticator = IAMAuthenticator('uguHWYx5mrhxXpFyOT2oFyZH1w50C0xl9-6mu1gj52U3')
speech_to_text = SpeechToTextV1(authenticator=stt_authenticator)
speech_to_text.set_service_url('https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/129f9103-7dc2-4ee7-a968-e97a869a66b4')

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    # Save the audio to a WAV file
    with open("input.wav", "wb") as f:
        f.write(audio.get_wav_data())

    # Use IBM Watson to transcribe the audio file
    with open("input.wav", "rb") as audio_file:
        response = speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/wav'
        ).get_result()
    
    try:
        text = response['results'][0]['alternatives'][0]['transcript']
        print(f"Recognized Text: {text}")
        return text
    except (IndexError, KeyError):
        print("Could not transcribe audio")
        return ""

def query_gpt3(prompt):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def speak(text):
    response = text_to_speech.synthesize(
        text,
        voice='en-US_AllisonV3Voice',
        accept='audio/wav'
    ).get_result()
    audio = response.content

    # Play the audio
    with open('response.wav', 'wb') as audio_file:
        audio_file.write(audio)

    chunk = 1024
    wf = wave.open('response.wav', 'rb')
    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(chunk)

    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()

    p.terminate()

if __name__ == "__main__":
    while True:
        text = recognize_speech()
        if text:
            response = query_gpt3(text)
            print(f"GPT-3 Response: {response}")
            speak(response)