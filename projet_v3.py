import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Charger l'audio
audio_path = '/home/morlot/code/numendo/essais/FULL MATCH Manchester City 2-1 Manchester United FINAL Emira.mp3'
audio, sr = librosa.load(audio_path, sr=None)

# Longueur de chaque segment (5 minutes)
segment_length = 5 * 60 * sr

# Découpage de l'audio en segments de 5 minutes
segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]

# Calcul de la moyenne du bruit pour chaque segment de 5 minutes
mean_amplitudes = [np.mean(np.abs(segment)) for segment in segments]

# Seuil pour les pics (basé sur la courbe lissée précédente)
threshold = np.percentile(gaussian_filter1d(np.abs(audio), sigma=50), 70)

# Identification des segments contenant des pics
pic_segments = [i for i, val in enumerate(mean_amplitudes) if val > threshold]

# Affichage de la courbe
plt.plot(mean_amplitudes)
plt.axhline(y=threshold, color='r', linestyle='--')
plt.title('Moyenne du bruit par segment de 5 minutes')
plt.xlabel('Segment (5 minutes)')
plt.ylabel('Moyenne du bruit')
plt.show()

# Récupération des intervalles de temps des pics
pic_intervals = [(i * 5, (i + 1) * 5) for i in pic_segments]
print("Intervalles de temps des pics (en minutes) :", pic_intervals)

from moviepy.editor import VideoFileClip, concatenate_videoclips

# Chemin vers la vidéo
video_path = '/home/morlot/code/numendo/essais/FULL MATCH Manchester City 2-1 Manchester United FINAL Emira.mkv'

# Charger la vidéo
video = VideoFileClip(video_path)

# Intervalles de temps des pics (en minutes), de votre code précédent
pic_intervals = [(i * 5, (i + 1) * 5) for i in pic_segments]

# Convertir les intervalles de temps en secondes pour la vidéo
pic_intervals_in_seconds = [(start * 60, end * 60) for start, end in pic_intervals]

# Extraire les clips correspondant aux intervalles de temps
clips = [video.subclip(start, end) for start, end in pic_intervals_in_seconds]

# Concaténer les clips
final_clip = concatenate_videoclips(clips)

# Exporter la vidéo finale
output_path = '/home/morlot/code/numendo/essais/resume.mkv'
final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
