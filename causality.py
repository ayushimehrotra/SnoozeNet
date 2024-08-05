import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# Granger causality is used to find the parts of the EEG epoch that are causally related 

# Function to perform Granger causality test on each window
def perform_granger_test(eeg_window, target_window, max_lag=5):
    data = np.vstack([target_window, eeg_window]).T
    df = pd.DataFrame(data, columns=['target', 'eeg'])
    result = grangercausalitytests(df, max_lag, verbose=False)
    p_values = [round(result[i+1][0]['ssr_chi2test'][1], 4) for i in range(max_lag)]
    min_p_value = np.min(p_values)
    return min_p_value

def causality_eeg_1(eeg_data, window_size, top_k):
    '''
        eeg_data: (batch_size, num_of_epochs, len_of_epochs, dimension_of_input)
        window_size: length of window to compare granger causality
        top_k: number of windows desired for output
    '''
    n_windows = epoch_length // window_size

    significant_windows = []
    p_value_threshold = 0.05  # Threshold for significance

for epoch in range(n_epochs):
    for window in range(n_windows):
        start_idx = window * window_size
        end_idx = start_idx + window_size
        eeg_window = eeg_data[epoch, start_idx:end_idx]
        target_window = target_data[epoch, start_idx:end_idx]
        
        p_value = perform_granger_test(eeg_window, target_window)
        if p_value < p_value_threshold:
            significant_windows.append((epoch, start_idx, end_idx))

# Aggregate selected windows to form reduced time dimension representation
# This step can vary based on specific requirements, such as averaging the selected windows

reduced_eeg_data = np.zeros((n_epochs, len(significant_windows)))
for i, (epoch, start_idx, end_idx) in enumerate(significant_windows):
    reduced_eeg_data[epoch, i] = np.mean(eeg_data[epoch, start_idx:end_idx])

print("Reduced EEG data shape:", reduced_eeg_data.shape)
