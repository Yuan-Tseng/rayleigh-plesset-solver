import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure
from scipy import ndimage
import pandas as pd
import os
import find_bubble

def auto_canny(image, sigma=0.33):
    # Calculate the median of the pixel intensities
    v = np.median(image)
    # Determine lower and upper thresholds for Canny edge detection based on sigma
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def extract_radius_from_video(video_path, fps, pixel_scale, show_video=False, roi=None, save_process_video=True):
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

    # === Prepare visualization ===
    video_writer = None
    output_video_path = os.path.splitext(video_path)[0] + '_processed.avi' 
    # ==========================

    while True:
        ret, frame = cap.read() # Read frame
        if not ret:             # if no frame is returned, end of video -> break
            break  

        frame = frame[roi_y : roi_y+roi_h, roi_x:roi_x+roi_w]

        # === Set video size ===
        if save_process_video and video_writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
            video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height)) # 30 FPS for output video
            print(f"Outputting Video: {output_video_path}")
        # =============================================================
        
        # 2. Pre-processing
        # Convert to grayscale (Canny works on single channel)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        brightness_threshold = 10 #62 
        
        if mean_brightness < brightness_threshold:
            print(f"Frame {frame_count} is too dark (Brightness: {mean_brightness:.2f}). Stopping extraction.")
            break  # Stop processing further frames
        
        # Optional: Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Edge Detection (Canny)
        # Thresholds (50, 150) might need tuning depending on contrast
        edges = auto_canny(blurred)
        # edges = cv2.Canny(gray, threshold1=30, threshold2=100)
        
        # 4. Morphological Operations
        # Dilate/Close to connect gaps in the edge
        # We convert CV2 edges (0-255) to boolean for skimage
        edges_bool = edges > 0
        dilated = morphology.binary_dilation(edges_bool, morphology.disk(2))
        
        # Fill holes to make a solid mask
        filled_mask = ndimage.binary_fill_holes(dilated)
        final_mask = morphology.binary_erosion(filled_mask, morphology.disk(2))
        
        # Remove small noise (artifacts)
        clean_mask = morphology.remove_small_objects(final_mask, min_size=50)
        
        # 5. Measurement [cite: 50]
        # Label the regions (should be only 1 bubble)
        label_img = measure.label(clean_mask)
        props = measure.regionprops(label_img)
        
        # Prepare display image
        display_img = (clean_mask * 255).astype(np.uint8)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

        if len(props) > 0:
            # Pick the largest object
            largest_bubble = max(props, key=lambda x: x.area)
            area_pixels = largest_bubble.area
            
            # Calculate Radius
            radius_pixels = np.sqrt(area_pixels / np.pi)
            radius_um = radius_pixels * pixel_scale
            
            # Safety check for abnormal radius drop
            if len(radius_time_series) > 0:
                initial_R = radius_time_series[0]
                
                # If radius drops below 60% of initial size, assume tracking lost
                if radius_um < initial_R * 0.6:
                    print(f"Radius dropped abnormally to {radius_um:.2f} um. Tracking likely lost. Stopping.")
                    break

            radius_time_series.append(radius_um)

            if show_video:
                centroid = largest_bubble.centroid
                # Draw circle on display image, for visualization (y,x)
                cv2.circle(display_img, (int(centroid[1]), int(centroid[0])), 2, (0, 0, 255), -1)
                # (Optional) Write radius on the image
                cv2.putText(display_img, f"R: {radius_um:.1f} um", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            radius_time_series.append(np.nan)

        frames.append(frame_count)
        frame_count += 1

        # Visualization
        if show_video:
            cv2.imshow('Processed Mask', display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        if save_process_video and video_writer is not None:
            video_writer.write(display_img)

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print("Video saved.")

    cv2.destroyAllWindows()
    
    # 6. Create Time Axis
    time_array = np.array(frames) / fps  # [seconds]
    radius_array = np.array(radius_time_series) # [um]
    
    return time_array, radius_array

# ================= Usage Example ================= 
# You MUST find these two values from your experiment notes/TAs!
FPS = 1e7             # Example: 10 Million FPS (Check your settings!)
SCALE = 0.2           # Example: 0.2 um per pixel (Check calibration!)

video_file = 'Data/AVI24/Camera_15_16_28/Camera_15_16_28.avi'
# 'Data/AVI24/Camera_15_02_48/Camera_15_02_48.avi'
# 'Data/AVI24/Camera_15_08_33/Camera_15_08_33.avi'
# 'Data/AVI24/Camera_15_16_28/Camera_15_16_28.avi'
# 'Data/AVI24/Camera_15_21_51/Camera_15_21_51.avi'

# Run processing
t, r = extract_radius_from_video(video_file, 
                                 FPS, 
                                 SCALE, 
                                 show_video=True, 
                                 roi=find_bubble.select_bubble_roi(video_file),
                                 save_process_video=True)
                                 

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
print("=========================================")
print(f"CSV SAVED! Radius data saved to {output_csv}")
print(f"Total frames processed: {len(df)}; Valid frames: {len(df_clean)}; Time Priod: {df_clean['Time_s'].iloc[-1]:.8f} s")
print(f"Initial Radius: {df_clean['Radius_um'].iloc[0]:.2f} um; Max Radius: {df_clean['Radius_um'].max():.2f} um")