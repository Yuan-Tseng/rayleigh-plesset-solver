# Microbubble Dynamics Analysis
**Course:** Hydrodynamics and Cavitation (HS2025), ETH Zurich

## 1. Project Overview
This repository contains the analysis pipeline for the laboratory session on ultrasonically driven, lipid-coated microbubbles. The project involves extracting radial dynamics from ultra-high-speed imaging and fitting the data to the **Rayleigh-Plesset equation** (incorporating the **Marmottant shell model**) to determine shell viscoelastic properties.

## 2. Experimental Setup & Parameters

### Key Parameters (Crucial for Analysis)
These values must be updated in the Python scripts (`FPS`, `SCALE`, `Sensitivity`).

* **Frame Rate (FPS):** `10,000,000` (10 Million FPS, i.e., 0.1 $\mu s$ per frame).
* **Pixel Scale:** `0.16` $\mu m$/pixel (FOV: $64 \times 40 \mu m$).
* **Hydrophone Sensitivity:** `39 mV/MPa` (or $3.9 \times 10^{-8}$ V/Pa).
* **Sample Setup:** Microbubbles are contained between transparent sheets (acoustically and optically transparent) to prevent floating while allowing observation.

### Trigger & Timing Sequence (The "Delayer")
Understanding the timing is essential to interpret the video start time and flash duration.
1.  **$t = 0~\mu s$:** Transducer triggered.
2.  **$t \approx 53\text{-}54~\mu s$:** **1st Flashlight** activates.
3.  **$t = 55~\mu s$:** **Camera** starts recording.
4.  **$t \approx 56~\mu s$:** Acoustic wave reaches the bubble (Distance $\approx 85$ mm / Speed of Sound in water).
5.  **$t \approx 60~\mu s$:** **2nd Flashlight** activates.

*Note: Dark frames at the end of the video are due to the flashlights fading out.*

## 3. Data Processing Workflow

### A. Signal Processing (Hydrophone)
* **Input File:** Use the **`F1--*.csv`** files (Filtered/Averaged signal), *not* the `C1` raw signal.
* **Units:** Time in **seconds**, Amplitude in **Volts**.
* **Expected Amplitude:** Approximately **2-3 mV** peak-to-peak.
* **Action:** Convert Volts to Pressure (Pa) using the sensitivity ($39$ mV/MPa) to derive the Driving Pressure $P_A$ and Frequency $f$.

### B. Image Processing (Bubble Radius)
* **Initial Radius ($R_0$):** Determined by analyzing the **first frame** of the video.
* **Region of Interest (Time):** The analysis should focus **only on the stably fluctuating interval**.
    * Typically, there are only **7-8 stable cycles** of fluctuation.
    * Data outside this range (start-up or post-flash noise) should be truncated.
* **Method:** Thresholding/Otsu's method is recommended over Canny edge detection due to optical halo effects.

### C. Simulation & Fitting
* **Model:** Rayleigh-Plesset equation with Marmottant shell terms.
* **Optimization Strategy:** **Trial and Error**.
    * Manually adjust Shell Elasticity ($\chi$) and Shell Viscosity ($\kappa_s$) in the simulation script.
    * **Goal:** Visually match the phase and amplitude of the simulated curve ($R_{sim}$) with the experimental curve ($R_{exp}$) for the stable cycles.

## 4. Features
1.  **Signal Processing:** Analyzes hydrophone data (`.csv`) to determine the actual driving frequency ($f$) and acoustic pressure amplitude ($P_A$).
2.  **Image Processing:** Extracts the radius-time curve $R(t)$ from high-speed video footage (`.avi`) using Canny edge detection and morphological operations. Includes automatic ROI selection and brightness safety checks.
3.  **Numerical Simulation:** Solves the Rayleigh-Plesset equation with the Marmottant model using `scipy.integrate.solve_ivp`.
4.  **Parameter Fitting:** Allows for manual optimization of shell viscosity ($\kappa_s$) and shell elasticity ($\chi$) to match experimental data.

## 5. Usage
1.  find_bubble.py: crop video and make grid picture
2.  Hydrophone.py: process the csv of hydrophone signal
3.  Image_process.py: process experimental date
4.  Rayleigh_Plesset.py: solve the equation numerically based on Rayleigh-Plesset and Marmottant model

### Step 1: Analyze Signal
```bash
# 1. Create a new environment
conda create -n bubble_lab python=3.9
conda activate bubble_lab

# 2. Install required packages (using conda-forge for best compatibility)
conda install -c conda-forge numpy scipy matplotlib pandas opencv scikit-image

