import librosa

audio_path = librosa.util.example_audio_file()
y, sr = librosa.load(audio_path)
