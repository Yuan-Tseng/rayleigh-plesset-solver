# Microbubble Dynamics Analysis
**Course:** Hydrodynamics and Cavitation (HS2025), ETH Zurich

## Project Overview
This repository contains the Python analysis pipeline for the laboratory session on ultrasonically driven, lipid-coated microbubbles. The goal is to investigate the radial dynamics of microbubbles through ultra-high-speed imaging and compare the experimental results with the theoretical Rayleigh-Plesset equation incorporated with the Marmottant shell model.

## Features
1.  **Signal Processing:** Analyzes hydrophone data (`.csv`) to determine the actual driving frequency ($f$) and acoustic pressure amplitude ($P_A$).
2.  **Image Processing:** Extracts the radius-time curve $R(t)$ from high-speed video footage (`.avi`) using Canny edge detection and morphological operations. Includes automatic ROI selection and brightness safety checks.
3.  **Numerical Simulation:** Solves the Rayleigh-Plesset equation with the Marmottant model using `scipy.integrate.solve_ivp`.
4.  **Parameter Fitting:** Allows for manual optimization of shell viscosity ($\kappa_s$) and shell elasticity ($\chi$) to match experimental data.

## Questions
1. How is the initial R determined through image processing? Current R is hard to define
2. Which interval of the video should be considered? Only the stably fluctuating one? But there's only 7-8 cycles of fluctuation. 
3. What is the fps and the pixel size? 10M fps, pixel size unknown. 
4. How should the csv be processed? What is the unit of it? 
5. Next step: How can I optimize chi and kappa? 

## Installation

It is recommended to use Conda to manage dependencies.

```bash
# 1. Create a new environment
conda create -n bubble_lab python=3.9
conda activate bubble_lab

# 2. Install required packages (using conda-forge for best compatibility)
conda install -c conda-forge numpy scipy matplotlib pandas opencv scikit-image

