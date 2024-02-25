"""
Version 1: Assuming annotations are given correctly, we are able to use them to extract all the features we want pretty easily.
For this sample patient, unfortunately, the annotations do not span the entire study, so we get a very skewed understanding of the metrics.
Regardless, this code works, and is a fairly simple feature extraction algorithm given the annotations. The problem is that annotations will not always be present.
"""
import mne as edf
import numpy as np
from scipy.signal import welch
from collections import Counter

# Load the PSG data and corresponding hypnogram (annotations) from files.
psg = edf.io.read_raw_edf('D:\FLOW\PSG\TestPatientPSG.edf', preload=True)
hypnogram = edf.read_annotations('D:\FLOW\PSG\TestPatientHypnogram.edf')

# Apply annotations to the PSG data to correlate EEG data with sleep stages.
psg.set_annotations(hypnogram, emit_warning=False)

# Apply a band-pass filter to the EEG channels to focus on frequencies between 0.5 and 30 Hz,
# covering delta, theta, alpha, and beta bands important for sleep analysis.
psg.filter(0.5, 30, picks='eeg', fir_design='firwin')

# Segment the filtered EEG data into 30-second epochs without overlap,
# conforming to the standard duration used for sleep stage analysis.
epochs = edf.make_fixed_length_epochs(psg, duration=30, overlap=0, preload=True)

# Function to extract Power Spectral Density (PSD) features from epochs,
# calculating average power within delta, theta, alpha, and beta frequency bands.
def extract_psd_features(epochs):
    features = []
    for epoch in epochs.get_data(copy=True):
        psd_list = []
        for channel_data in epoch:
            psd, freq = welch(channel_data, nperseg=channel_data.shape[0], fs=epochs.info['sfreq'])
            # Compute mean power for each frequency band, handling cases without significant power.
            delta = psd[(freq >= 0.5) & (freq < 4)].mean() if np.any((freq >= 0.5) & (freq < 4)) else 0
            theta = psd[(freq >= 4) & (freq < 8)].mean() if np.any((freq >= 4) & (freq < 8)) else 0
            alpha = psd[(freq >= 8) & (freq < 13)].mean() if np.any((freq >= 8) & (freq < 13)) else 0
            beta = psd[(freq >= 13) & (freq < 30)].mean() if np.any((freq >= 13) & (freq < 30)) else 0
            psd_list.append([delta, theta, alpha, beta])
        features.append(np.mean(psd_list, axis=0))
    return np.array(features)

# Extract features for sleep stage analysis.
features = extract_psd_features(epochs)

# Mapping of annotation descriptions to concise sleep stage labels.
stage_mapping = {
    'Sleep stage W': 'Awake',
    'Sleep stage 1': 'N1',
    'Sleep stage 2': 'N2',
    'Sleep stage 3': 'N3',
    'Sleep stage 4': 'N3',  # Combining stages 3 and 4 as N3.
    'Sleep stage R': 'REM',
    'Sleep stage ?': 'Unknown'
}

# Map sleep stage descriptions from annotations to standardized labels.
mapped_sleep_stages = [stage_mapping.get(desc, 'Unknown') for desc in psg.annotations.description]

# Count epochs per sleep stage and compute time spent in each stage.
stage_counts = Counter(mapped_sleep_stages)
stage_times = {stage: count * 0.5 for stage, count in stage_counts.items()}  # Convert epoch count to minutes.

# Calculate total sleep time excluding 'Awake' and 'Unknown' stages and total awake time.
total_sleep_time = sum(time for stage, time in stage_times.items() if stage not in ['Awake', 'Unknown'])
total_awake_time = stage_times.get('Awake', 0)

# Print the computed metrics with visual separators for clarity.
print("_________________________________________________________")
print("")
print("Time spent in each sleep stage (minutes):", stage_times)
print("_________________________________________________________")
print("")
print("Total Sleep Time (minutes):", total_sleep_time)
print("_________________________________________________________")
print("")
print("Total Awake Time (minutes):", total_awake_time)
print("_________________________________________________________")
print("")
print(f"Extracted features shape: {features.shape}")
print("_________________________________________________________")
print("")
