import cv2
import os
import shutil
import numpy as np
import glob

# ================= Settings =================
# 1. Folder for input images
"""
Data/AVI24/Camera_15_02_48/Camera_15_02_48_cropped_picture
Data/AVI24/Camera_15_08_33/Camera_15_08_33_cropped_picture
Data/AVI24/Camera_15_16_28/Camera_15_16_28_cropped_picture
Data/AVI24/Camera_15_21_51/Camera_15_21_51_cropped_picture
"""

INPUT_FOLDER = "Data/AVI24/Camera_15_21_51/Camera_15_21_51_cropped_picture"
OUTPUT_FILENAME = "Bubble_Evolution_Grid.png"

FPS = 1e6             # 1 Million FPS (1 us per frame)
START_TIME_US = 30    
TIME_OFFSET = 0      

# 2. Grid settings for output image
ROWS = 4
COLS = 10
SKIP_FRAMES = 1       # Take every 2nd frame
# ===========================================

def select_bubble_roi(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        print("Cannot read video file.")
    else:
        print("Select the bubble region by drawing a rectangle. Press ENTER or SPACE to confirm.")
        print("Press C to cancel selection.")

        roi = cv2.selectROI("Select Bubble", frame, showCrosshair=True, fromCenter=False)
        
        # Close the window and release the video capture
        cv2.destroyAllWindows()
        cap.release()

        # roi (x, y, w, h)
        x, y, w, h = roi
        
        if w > 0 and h > 0:
            print("\nResults of ROI Selection:")
            print(f"ROI coordinates: x={x}, y={y}, w={w}, h={h}")
            print("================================")
            print(f"Put image processing codes: frame = frame[{y}:{y+h}, {x}:{x+w}]")
        else:
            print("You did not select a valid ROI.")

    return roi

def select_bubble_center(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    w = 96
    h = 96

    if not ret:
        print("Cannot read video file.")
    else:
        print("Select the bubble region by drawing a rectangle. Press ENTER or SPACE to confirm.")
        print("Press C to cancel selection.")

        roi = cv2.selectROI("Select Bubble", frame, showCrosshair=True, fromCenter=False)
        
        # Close the window and release the video capture
        cv2.destroyAllWindows()
        cap.release()

        if roi is not None:
            x, y, width, height = roi
            center_x = x + width / 2
            center_y = y + height / 2
            x = int(center_x - w // 2)
            y = int(center_y - h // 2)
            if x < 0: x = 0
            if y < 0: y = 0
            print("\nResults of Center Selection:")
            print(f"Center coordinates: x={center_x}, y={center_y}")
            print("================================")
            final_roi = (x, y, w, h)
            print(f"Computed ROI from center: x={final_roi[0]}, y={final_roi[1]}, w={final_roi[2]}, h={final_roi[3]}")
            return final_roi
        else:
            print("You did not select a valid center.")

    return None

def select_and_save_cropped(video_path):
    """
    After selecting the bubble ROI, this function crops the video accordingly and saves:
    1. Cropped video to a new .avi file (Overwrite)
    2. Cropped frames as individual .png images in a new folder (Overwrite)
    """
    
    # 1. Choose ROI
    roi = select_bubble_center(video_path)
    if roi is not None:
        x, y, w, h = roi
    else:
        print("No ROI selected, exiting function.")
        return

    # 2. Prepare output paths
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    directory = os.path.dirname(video_path)
    output_video_path = os.path.join(directory, f"{base_name}_cropped.avi")
    output_img_dir = os.path.join(directory, f"{base_name}_cropped_picture")

    # 3. Overwrite existing output folder if exists
    if os.path.exists(output_img_dir):
        print(f"Folder Exists, Overwriting Old Contents: {output_img_dir}")
        shutil.rmtree(output_img_dir)
    
    os.makedirs(output_img_dir) # Create new folder
    print(f"New Folder Created: {output_img_dir}")

    # 4. Process video: crop and save
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 # Default FPS if cannot be read

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    print(f"Processing and Saving to: {output_video_path}")
    
    frame_count = 0
    while True and frame_count < 160:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop frame
        cropped_frame = frame[y:y+h, x:x+w]

        # A. Write to cropped video
        out.write(cropped_frame)

        # B. Write to individual image files
        # Camera_15_02_48_0.png, Camera_15_02_48_1.png ...
        img_name = f"{base_name}_{frame_count}.png"
        img_path = os.path.join(output_img_dir, img_name)
        cv2.imwrite(img_path, cropped_frame)
        
        frame_count += 1

    cap.release()
    out.release()
    print(f"\nSaved {frame_count} Frames")

def create_bubble_grid():
    # 1. read all .png files in the input folder and sort them
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.png"))
    
    if not files:
        print(f"Error: {INPUT_FOLDER} no PNG files found.")
        return

    # Logic to sort files based on the numeric part after the last underscore
    try:
        files.sort(key=lambda f: int(os.path.splitext(f)[0].split('_')[-1]))
    except ValueError:
        print("Warning: Filename format is incorrect for sorting. Ensure filenames end with an underscore followed by a number.")
        files.sort()

    # 2. Sample files according to SKIP_FRAMES
    sampled_files = files[::SKIP_FRAMES]
    
    # Only need ROWS * COLS images
    total_images_needed = ROWS * COLS
    if len(sampled_files) < total_images_needed:
        print(f"Warning: {total_images_needed} images needed, not enough images in folder")
    else:
        sampled_files = sampled_files[:total_images_needed]

    # 3. Process each image: add time label and border
    processed_images = []
    ref_img = cv2.imread(sampled_files[0])
    h, w, _ = ref_img.shape
    
    # TEXT settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7 
    thickness = 1
    color = (0, 0, 0) # Black color
    
    for i, file_path in enumerate(sampled_files):
        img = cv2.imread(file_path)
        if img is None: continue
        
        # Ensure same size
        if img.shape != (h, w, 3):
            img = cv2.resize(img, (w, h))
            
        # Time calculation
        # Original frame index = i * SKIP_FRAMES
        # Time = Index / FPS * 1e6 (us)
        frame_idx = i * SKIP_FRAMES
        time_us = START_TIME_US + (frame_idx / FPS) * 1e6 + TIME_OFFSET
        
        label_text = f"t={int(time_us)}us"
        cv2.putText(img, label_text, (2, 17), font, font_scale, color, thickness, cv2.LINE_AA)
        # Add border
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        processed_images.append(img)

    # 4. If not enough images, fill with blank
    while len(processed_images) < total_images_needed:
        blank = np.zeros((h+2, w+2, 3), dtype=np.uint8) # +2 for border
        processed_images.append(blank)

    # 5. Create grid
    rows_img = []
    for r in range(ROWS):
        # This row of images 
        row_imgs = processed_images[r*COLS : (r+1)*COLS]
        # Horizontal concatenation
        row_concat = np.hstack(row_imgs)
        rows_img.append(row_concat)
    
    # Vertical concatenation
    final_grid = np.vstack(rows_img)
    
    # 6. Save final grid image
    cv2.imwrite(OUTPUT_FILENAME, final_grid)
    print(f"Grid Generation Successful: {OUTPUT_FILENAME}")

if __name__ == "__main__":

    reselect = False
    if reselect:
        for video_file in [
            'Data/AVI24/Camera_15_02_48/Camera_15_02_48.avi',
            'Data/AVI24/Camera_15_08_33/Camera_15_08_33.avi',
            'Data/AVI24/Camera_15_16_28/Camera_15_16_28.avi',
            'Data/AVI24/Camera_15_21_51/Camera_15_21_51.avi'
        ]:
            if os.path.exists(video_file):
                select_and_save_cropped(video_file)
            else:
                print(f"File not found: {video_file}")
    
    create_grid = True
    if create_grid:
        create_bubble_grid()
    
