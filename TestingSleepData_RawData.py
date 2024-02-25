"""
Version 2: This version will assume that the annotations are not given. Now, the task becomes to actually extract correct data directly from EDF.
This is very difficult in comparison, but more accurately reflects the true algorithm we will likely need. This is my best attemt with the time I had but not perfect.    
"""
import mne as edf
import numpy as np
from scipy.signal import welch
from sklearn.cluster import KMeans
from collections import Counter

# Load the polysomnography (PSG) data from a specified file.
psg = edf.io.read_raw_edf('D:\FLOW\PSG\TestPatient2PSG.edf', preload=True)

# Apply a band-pass filter to EEG channels to isolate the frequency range of interest (0.5-30 Hz),
# which includes delta, theta, alpha, and beta bands relevant to sleep analysis.
psg.filter(0.5, 30, picks='eeg', fir_design='firwin')

# Segment the continuous EEG signal into 30-second epochs with no overlap between them.
# This standard epoch length is used for analyzing sleep stages.
epochs = edf.make_fixed_length_epochs(psg, duration=30, overlap=0, preload=True)

# Define a function to extract relevant features from each epoch for the purpose of clustering.
# Features include power spectral density (PSD) values across delta, theta, alpha, and beta frequency bands.
def extract_features_for_clustering(epochs):
    features = []
    for epoch in epochs.get_data(copy=True):
        psd_features = []
        for channel_data in epoch:
            freqs, psd = welch(channel_data, fs=epochs.info['sfreq'])
            # Calculate the mean PSD within specified frequency bands for each channel.
            delta = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
            theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
            alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)])
            beta = np.mean(psd[(freqs >= 13) & (freqs < 30)])
            # Aggregate the PSD features for clustering.
            psd_features.extend([delta, theta, alpha, beta])
        features.append(psd_features)
    return np.array(features)

# Extract features from each epoch, making them suitable for clustering analysis.
features_for_clustering = extract_features_for_clustering(epochs)

# Implement a function to classify epochs into sleep stages using the K-means algorithm,
# an unsupervised learning approach, based on the extracted features.
def classify_sleep_stages(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    return kmeans.labels_

# Apply the classification function to group epochs into clusters, potentially representing different sleep stages.
sleep_stages_clusters = classify_sleep_stages(features_for_clustering)

# Count the number of epochs assigned to each cluster by the K-means classification.
epoch_cluster_counts = Counter(sleep_stages_clusters)

# Map the cluster labels to hypothetical sleep stages based on a predefined mapping.
# This mapping is speculative and should be refined with further analysis or validation.
cluster_to_stage_mapping = {
    0: 'N3',  # Deep, restorative sleep
    1: 'N2',  # Most common, light sleep
    2: 'Awake', # Periods of wakefulness
    3: 'N1',  # Transitional, lightest sleep
    4: 'REM'   # Rapid Eye Movement sleep, associated with dreaming
}

# Translate the cluster labels into sleep stages using the predefined mapping.
sleep_stages = [cluster_to_stage_mapping[cluster] for cluster in sleep_stages_clusters]

# Count the number of epochs in each translated sleep stage.
sleep_stage_counts = Counter(sleep_stages)

# Convert the epoch counts to total time spent in each sleep stage, both in minutes.
sleep_stage_minutes = {stage: count * 0.5 for stage, count in sleep_stage_counts.items()}

# Convert the epoch counts to total time spent in each sleep stage, in hours, and round to 2 decimal points.
sleep_stage_hours = {stage: round(count * (0.5/60), 2) for stage, count in sleep_stage_counts.items()}

# Print the total time spent in each sleep stage, presented in minutes.
print("_________________________________________________________")
print("")
print("Time spent in each sleep stage (minutes):", sleep_stage_minutes)

# Print the total time spent in each sleep stage, presented in hours, with values rounded to 2 decimal points.
print("_________________________________________________________")
print("")
print("Time spent in each sleep stage (hours):", sleep_stage_hours)
print("_________________________________________________________")
print("")
