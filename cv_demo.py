import cv2
import os


for filename in sorted(os.listdir("clips")):

    cap = cv2.VideoCapture(f"clips/{filename}")
    print(f"clips/{filename}")
    if (cap.isOpened() == False):
        print("Error")
    
    while(cap.isOpened()):
    
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        # Display the resulting frame
            cv2.imshow('Frame', frame)
            
        # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # Break the loop
        else:
            break

    cap.release()
cv2.destroyAllWindows()