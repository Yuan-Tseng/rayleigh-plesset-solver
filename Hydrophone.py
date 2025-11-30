import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sensitivity = 39 # Hydrophone sensitivity [mV/MPa]
csv_file = 'Data/C1--1.5mhz-100mV-4.csv'
# 'Data/F1--1.5mhz-100mV-4.csv'

def process_hydrophone_data(csv_file, sensitivity):
    """
    Process hydrophone CSV data to extract pressure signal and find peaks.
    
    Returns:
    - time: Time array in seconds
    - pressure: Pressure array in MPa
    - peak_times: Times of detected peaks in seconds
    - peak_pressures: Pressures at detected peaks in MPa
    """
    # Load data
    data = pd.read_csv(csv_file)
    
    # Assuming the CSV has 'Time' and 'Voltage' columns
    time = data['Time'].values  # in seconds
    voltage = data['Voltage'].values  # in mV
    
    # Convert voltage to pressure (MPa)
    pressure = voltage / sensitivity  # in MPa
    
    # Find peaks in the pressure signal
    peaks, _ = find_peaks(pressure, height=0)  # Only consider positive peaks
    
    peak_times = time[peaks]
    peak_pressures = pressure[peaks]
    
    return time, pressure, peak_times, peak_pressures   
