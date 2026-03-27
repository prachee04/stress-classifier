import numpy as np

def create_windows(chest_data, labels, window_size=60, sampling_rate=700):
    
    samples_per_window = window_size * sampling_rate   #42000 samples
    
    windows = []
    
    for start in range(0, len(labels) - samples_per_window, samples_per_window):
        
        end = start + samples_per_window
        
        label_window = labels[start:end]
        
        # Keep only clean windows (same label)
        if len(set(label_window)) == 1:
            
            window = {
                "ECG": chest_data['ECG'][start:end],
                "EDA": chest_data['EDA'][start:end],
                "EMG": chest_data['EMG'][start:end],
                "Resp": chest_data['Resp'][start:end],
                "Temp": chest_data['Temp'][start:end],
                "ACC": chest_data['ACC'][start:end],
                "Label": label_window[0]
            }
            
            windows.append(window)
    
    return windows