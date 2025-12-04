"""
This module processes hydrophone data from CSV files to extract pressure signals
This is the input for the Rayleigh-Plesset Solver.

The original unit:
column D [seconds]
column E [V]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

SENSITIVITY = 39 # Hydrophone sensitivity [mV/MPa]
CSV = 'Data/cleaned_F1--1.5mhz-100mV-4.csv' # Filtered data, recommended to use this one
# CSV = 'Data/C1--1.5mhz-100mV-4.csv' # Original data, unfiltered, so noisy
# CSV = 'DATA/F1--1.5mhz-100mV-4.csv'

DATA = pd.read_csv(CSV)
time = DATA.iloc[1:, 0].to_numpy()  # Time in seconds [:, 3] for uncleaned
voltage = DATA.iloc[1:, 1].to_numpy()  # Voltage in Volts [:, 4] for uncleaned 
cutoff_time = 55  # [us] It takes 55 us for sound wave to reach hydrophone (85mm away from transducer)

def process_hydrophone_data():
    """
    Process hydrophone CSV data to extract pressure signal and find peaks.
    
    Returns:
    - time: Time array in seconds
    - pressure: Pressure array in MPa
    - peak_times: Times of detected peaks in seconds
    - peak_pressures: Pressures at detected peaks in MPa
    """
    
    # Convert voltage to pressure (MPa)
    pressure = (voltage*10**3) / SENSITIVITY  # in MPa
    
    # Find peaks in the pressure signal
    peaks, _ = find_peaks(pressure, height=0)  # Only consider positive peaks
    
    peak_times = time[peaks]
    peak_pressures = pressure[peaks]
    
    return time, pressure, peak_times, peak_pressures

def plot_voltage_signal():
    """
    Plot the voltage signal and mark detected peaks.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time * 10**6, voltage * 10**3, label='Volatge (mV)')
    plt.xlabel('Time (us)')
    plt.ylabel('Volatge (mV)')
    plt.title('Hydrophone Voltage Signal with Detected Peaks')
    plt.legend()
    plt.grid()
    plt.show() 

def plot_pressure_signal(time, pressure, peak_times, peak_pressures):
    """
    Plot the pressure signal and mark detected peaks.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t*10**6, p*10**3, label='Pressure (kPa)')
    plt.xlabel('Time (us)')
    plt.ylabel('Pressure (kPa)')
    plt.title('Hydrophone Pressure Signal with Detected Peaks')
    plt.legend()
    plt.grid()
    plt.show()

def clean_csv_data(cutoff_time, cleaned_csv=False):
    output_csv = 'Data/cleaned_' + CSV.split('/')[-1]
    df = pd.DataFrame({'Time_s': time, 'Voltage_V': voltage})
    df_clean = df[df['Time_s'] >= cutoff_time * 1e-6]
    df_clean.to_csv(output_csv, index=False)
    print("=========================================")
    print(f"CSV SAVED! Cleaned data saved to {output_csv}")
    print(f"Total data points: {len(df)}; Cleaned data points: {len(df_clean)}; Time starts from: {df_clean['Time_s'].iloc[0]:.8f} s")

if __name__ == "__main__":
    # Process hydrophone data
    t, p, peak_t, peak_p = process_hydrophone_data()

    clean_csv = False
    # Plot the voltage signal with peaks
    plot_voltage_signal()
    # Plot the pressure signal with peaks
    plot_pressure_signal(t, p, peak_t, peak_p)
    # Clean and save CSV data
    if clean_csv:
        clean_csv_data(cutoff_time, cleaned_csv=True)
        
