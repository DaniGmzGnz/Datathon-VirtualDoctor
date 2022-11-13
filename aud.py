import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    print(voice.id)
    engine.setProperty('voice', voices[1].id)
    engine.say('The quick brown fox jumped over the lazy dog.')
engine.runAndWait()