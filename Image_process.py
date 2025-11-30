import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure
from scipy import ndimage
import pandas as pd
import os
import find_bubble

def extract_radius_from_video(video_path, fps, pixel_scale, show_video=False, roi=None):
    """
    Process high-speed video to extract bubble radius over time.
    
    Parameters:
    - video_path: Path to the .avi file
    - fps: Frames per second (from experiment settings)
    - pixel_scale: micrometers per pixel (um/px)
    - show_video: If True, show the processing window (slower)
    """
    
    # 1. Load Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None, None

    radius_time_series = []
    frames = []
    frame_count = 0

    roi_x, roi_y, roi_w, roi_h = (0, 0, 0, 0)
    if roi is not None:
        roi_x, roi_y, roi_w, roi_h = roi

    while True:
        ret, frame = cap.read() # Read frame
        if not ret:             # if no frame is returned, end of video -> break
            break  

        frame = frame[roi_y : roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # 2. Pre-processing
        # Convert to grayscale (Canny works on single channel)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        brightness_threshold = 62 
        
        if mean_brightness < brightness_threshold:
            print(f"Frame {frame_count} is too dark (Brightness: {mean_brightness:.2f}). Stopping extraction.")
            break  # Stop processing further frames
        
        # 3. Edge Detection (Canny)
        # Thresholds (50, 150) might need tuning depending on contrast
        edges = cv2.Canny(gray, threshold1=30, threshold2=100)
        
        # 4. Morphological Operations
        # Dilate/Close to connect gaps in the edge
        # We convert CV2 edges (0-255) to boolean for skimage
        edges_bool = edges > 0
        closed_edges = morphology.binary_closing(edges_bool, morphology.disk(3))
        
        # Fill holes to make a solid mask
        filled_mask = ndimage.binary_fill_holes(closed_edges)
        
        # Remove small noise (artifacts)
        clean_mask = morphology.remove_small_objects(filled_mask, min_size=100)
        
        # 5. Measurement [cite: 50]
        # Label the regions (should be only 1 bubble)
        label_img = measure.label(clean_mask)
        props = measure.regionprops(label_img)
        
        if len(props) > 0:
            # Pick the largest object (assuming it's the main bubble)
            largest_bubble = max(props, key=lambda x: x.area)
            area_pixels = largest_bubble.area
            
            # Calculate Radius: R = sqrt(Area / pi) 
            radius_pixels = np.sqrt(area_pixels / np.pi)
            
            # Convert to physical units (micrometers)
            radius_um = radius_pixels * pixel_scale
            radius_time_series.append(radius_um)
        else:
            # If detection fails, use NaN or previous value
            radius_time_series.append(np.nan)

        frames.append(frame_count)
        frame_count += 1

        # Visualization (Optional)
        if show_video:
            # Show the filled mask to check if processing works
            display = (clean_mask * 255).astype(np.uint8)
            cv2.imshow('Processed Mask', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    # 6. Create Time Axis
    time_array = np.array(frames) / fps  # [seconds]
    radius_array = np.array(radius_time_series) # [um]
    
    return time_array, radius_array

# ================= Usage Example ================= 
# You MUST find these two values from your experiment notes/TAs!
FPS = 1e6             # Example: 1 Million FPS (Check your settings!)
SCALE = 0.2           # Example: 0.2 um per pixel (Check calibration!)

video_file = 'Data/AVI24/Camera_15_02_48/Camera_15_02_48.avi'
# 'Data/AVI24/Camera_15_02_48/Camera_15_02_48.avi'
# 'Data/AVI24/Camera_15_08_33/Camera_15_08_33.avi'
# 'Data/AVI24/Camera_15_16_28/Camera_15_16_28.avi'
# 'Data/AVI24/Camera_15_21_51/Camera_15_21_51.avi'

# Run processing
t, r = extract_radius_from_video(video_file, FPS, SCALE, show_video=True, roi=find_bubble.select_bubble_roi(video_file))

# Quick Plot to check
plt.figure()
plt.plot(t * 1e6, r) # Time in microseconds
plt.xlabel('Time [us]')
plt.ylabel('Radius [um]')
plt.title('Experimental Bubble Radius')
plt.grid(True)
plt.show()

# Save to CSV
output_csv = os.path.splitext(video_file)[0] + '_radius.csv'
df = pd.DataFrame({'Time_s': t, 'Radius_um': r, 'Radius_m': r * 1e-6})
df_clean = df.dropna()
df_clean.to_csv(output_csv, index=False)
print(f"Radius data saved to {output_csv}")
print(f"Total frames processed: {len(df)}; Valid frames: {len(df_clean)}")
print(f"Initial Radius: {df_clean['Radius_um'].iloc[0]:.2f} um; Max Radius: {df_clean['Radius_um'].max():.2f} um")