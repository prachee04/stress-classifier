import numpy as np
from collections import Counter

def create_windows(chest_data, labels, window_size=60, sampling_rate=700):
    
    samples_per_window = window_size * sampling_rate
    windows = []
    
    VALID_LABELS = [1, 2, 3]

    for start in range(0, len(labels) - samples_per_window, samples_per_window):
    
        end = start + samples_per_window
        label_window = labels[start:end]
        
        label_counts = Counter(label_window)
        majority_label, count = label_counts.most_common(1)[0]

        if majority_label in VALID_LABELS and count / len(label_window) >= 0.8:
            
            window = {
                "ECG": chest_data['ECG'][start:end],
                "EDA": chest_data['EDA'][start:end],
                "EMG": chest_data['EMG'][start:end],
                "Resp": chest_data['Resp'][start:end],
                "Temp": chest_data['Temp'][start:end],
                "ACC": chest_data['ACC'][start:end],
                "Label": majority_label
            }
            
            windows.append(window)
    
    return windows