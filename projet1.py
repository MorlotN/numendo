import librosa

# Charger l'audio
audio_path = '/home/morlot/code/numendo/essais/FULL MATCH Manchester City 2-1 Manchester United FINAL Emira.mp3'
audio, sr = librosa.load(audio_path, sr=None)

import matplotlib.pyplot as plt

# Visualiser l'audio sous forme d'onde
# plt.figure(figsize=(14, 5))
import librosa.display
import numpy as np

# D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram')
# plt.show()

# librosa.display.waveplot(audio, sr=sr)
# plt.show()



plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio, sr=sr)
plt.title('Waveform')
plt.show()



# Exemple : Extraction de MFCCs
mfccs = librosa.feature.mfcc(y=audio, sr=sr)


from moviepy.editor import VideoFileClip

video_path = '/home/morlot/code/numendo/essais/FULL MATCH Manchester City 2-1 Manchester United FINAL Emira.mkv'
video = VideoFileClip(video_path)

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Charger l'audio
audio, sr = librosa.load(audio_path, sr=None)

# Calcul de l'amplitude
amplitude = np.abs(audio)

# Lissage avec une courbe de Gauss
smoothed_amplitude = gaussian_filter1d(amplitude, sigma=50)  # sigma à ajuster

# Seuil pour les 30% supérieurs
threshold = np.percentile(smoothed_amplitude, 70)

# Identification des périodes au-dessus du seuil
high_amplitude = smoothed_amplitude > threshold

# Visualisation
plt.figure(figsize=(14, 5))
plt.plot(smoothed_amplitude)
plt.axhline(y=threshold, color='r', linestyle='--')
plt.show()

# Vous pouvez maintenant isoler ces périodes pour une analyse plus approfondie

# Longueur de chaque segment (1 minute)
segment_length = 60 * sr

# Découpage de l'audio en segments
segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]

# Calcul de la moyenne du bruit pour chaque segment
mean_amplitudes = [np.mean(np.abs(segment)) for segment in segments]

# Affichage de la courbe
plt.plot(mean_amplitudes)
plt.xlabel('Temps (minutes)')
plt.ylabel('Moyenne du bruit')
plt.title('Moyenne du bruit par segment d\'une minute')
plt.show()
