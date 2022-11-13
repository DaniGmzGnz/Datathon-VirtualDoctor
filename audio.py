
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pyttsx3

model = whisper.load_model("base.en")


def get_audio_symptom(model, feature_names):

    engine = pyttsx3.init()

    #engine.setProperty('voice', "english+f5")
    #engine.setProperty('rate', 130)

    engine.say("Hello Dani, you must answer some short questions in order to obtain a prediction on a disease you may have.")
    engine.say("Please, tell me the most important symptom you are experiencing.")
    engine.runAndWait()
    engine.stop()

    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording

    print("Recording Audio")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('audios/init_symptom.wav', fs, myrecording)  # Save as WAV file 
    print("Audio Written")


    result = model.transcribe("audios/init_symptom.wav")
    print(result["text"])

    return result["text"]

get_audio_symptom(model)