import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Charger l'audio
audio_path = '/home/morlot/code/numendo/essais/FULL MATCH Manchester City 2-1 Manchester United FINAL Emira.mp3'
audio, sr = librosa.load(audio_path, sr=None)

# Longueur de chaque segment (1 minute)
segment_length = 1 * 60 * sr

# Découpage de l'audio en segments de 1 minute
segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]

# Calcul de la moyenne du bruit pour chaque segment de 1 minute
mean_amplitudes = [np.mean(np.abs(segment)) for segment in segments]

# Seuil pour les pics (basé sur la courbe lissée précédente)
threshold = np.percentile(gaussian_filter1d(np.abs(audio), sigma=50), 70)

# Identification des segments contenant des pics
pic_segments = [i for i, val in enumerate(mean_amplitudes) if val > threshold]

# Affichage de la courbe
plt.plot(mean_amplitudes)
plt.axhline(y=threshold, color='r', linestyle='--')
plt.title('Moyenne du bruit par segment de 1 minute')
plt.xlabel('Segment (1 minute)')
plt.ylabel('Moyenne du bruit')
plt.show()

# Récupération des intervalles de temps des pics
pic_intervals = [(i, i + 1) for i in pic_segments]
print("Intervalles de temps des pics (en minutes) :", pic_intervals)

# Chemin vers la vidéo
video_path = '/home/morlot/code/numendo/essais/FULL MATCH Manchester City 2-1 Manchester United FINAL Emira.mkv'
video = VideoFileClip(video_path)

# Convertir les intervalles de temps en secondes pour la vidéo
pic_intervals_in_seconds = [(start * 60, end * 60) for start, end in pic_intervals]

# Extraire les clips correspondant aux intervalles de temps
clips = [video.subclip(start, end) for start, end in pic_intervals_in_seconds]

# Concaténer les clips
final_clip = concatenate_videoclips(clips)

# Exporter la vidéo finale
output_path = '/home/morlot/code/numendo/essais/resume.mkv'
final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

##############################################################################
# Traitement des commentateurs
import speech_recognition as sr
from pydub import AudioSegment

recognizer = sr.Recognizer()
# audio_file = 'chemin/vers/votre/fichier/audio.mp3'


# Convertir le MP3 en WAV
# audio_path_mp3 = '/home/morlot/code/numendo/essais/FULL MATCH Manchester City 2-1 Manchester United FINAL Emira.mp3'
audio_path_wav = '/home/morlot/code/numendo/essais/FULL MATCH Manchester City 2-1 Manchester United FINAL Emira.wav'

sound = AudioSegment.from_mp3(audio_path)
sound.export(audio_path_wav, format="wav")

# Charger le fichier WAV
audio = AudioSegment.from_wav(audio_path_wav)

# Découper en segments (par exemple, segments de 30 secondes)
segment_length = 30 * 1000  # Durée en millisecondes
segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]

# Chemin du fichier texte où stocker les transcriptions
transcription_file_path = '/home/morlot/code/numendo/essais/transcriptions.txt'

with open(transcription_file_path, "w") as file:
    for i, segment in enumerate(segments):
        # Calculer le temps de début du segment en secondes
        start_time = i * segment_length / 1000  # Convertir en secondes

        segment.export("temp_segment.wav", format="wav")
        with sr.AudioFile("temp_segment.wav") as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data, language="en-US")
                print(f"Temps: {start_time}s - Segment {i}: {text}")
                file.write(f"Temps: {start_time}s - Segment {i}: {text}\n\n")
            except sr.RequestError as e:
                print(f"Erreur de requête pour le segment {i}: {e}")
                file.write(f"Erreur de requête pour le segment {i}: {e}\n\n")
            except sr.UnknownValueError:
                print(f"Aucune parole détectée dans le segment {i}")
                file.write(f"Aucune parole détectée dans le segment {i}\n\n")

print(f"Transcriptions sauvegardées dans {transcription_file_path}")
    


