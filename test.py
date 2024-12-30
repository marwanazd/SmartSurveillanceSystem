import cv2

def stream_camera():
    # Open a connection to the webcam
    camera = cv2.VideoCapture(0)  # Use 0 for the default camera, or adjust for other devices

    # Check if the camera is opened successfully
    if not camera.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Streaming video. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()

        if not ret:
            print("Error: Unable to capture video.")
            break

        # Display the frame in a window
        cv2.imshow('Video Stream', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_camera()
