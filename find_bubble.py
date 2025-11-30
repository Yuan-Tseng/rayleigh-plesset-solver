import cv2

# ============================= Bubble ROI Selection =============================#

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