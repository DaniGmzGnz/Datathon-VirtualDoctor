
import sounddevice as sd
from scipy.io.wavfile import write
import whisper

model = whisper.load_model("medium.en")

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('audios/init_symptom.wav', fs, myrecording)  # Save as WAV file 


result = model.transcribe("audios/init_symptom.wav")
print(result["text"])