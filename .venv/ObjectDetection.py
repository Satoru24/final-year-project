# Importing necessary libraries
import cv2
import numpy as np
import serial
import time

# Setting the detection threshold
thres = 0.45   # Threshold to detect object (confidence level)


# List to hold the class names from the COCO dataset
classNames= []

# Path to the file containing class labels (COCO dataset)
classFile = 'coco.names'

# Reading the class names from the file
try:
    with open(classFile,'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    print(f"Successfully loaded {len(classNames)} class names from {classFile}")
except Exception as e:
    print(f"Error loading class names: {e}")
    exit()


# Paths to the model configuration and pre-trained weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'


# Loading the pre-trained deep learning model using OpenCV's DNN module
try:
    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    print("Successfully loaded DNN the model")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()


# Setting input parameters for the model (required for SSD MobileNet)
net.setInputSize(320,320)   # Set input size for the model
net.setInputScale(1.0/ 127.5)  # Scale input pixels
net.setInputMean((127.5, 127.5, 127.5))  # Normalize input by subtracting mean
net.setInputSwapRB(True)   # Swap Red and Blue channels (OpenCV uses BGR by default)


# Function to detect objects in an image
def getObjects(img,draw=True,objects=[]):
    """
        Detects specified objects in the input image using the pre-trained model.

        Parameters:
        img: The input image/frame
        draw: Boolean flag to draw bounding boxes and labels
        objects: List of specific objects to detect

        Returns:
        img: Annotated image
        objectInfo: List of detected objects' info (bounding box, etc.)
        """

    # Perform object detection
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=0.2)

    # Default to detect all classes if specific objects are not specified
    if len(objects) == 0:
        objects = classNames

    objectInfo = []  # Initialize empty list to store detected object info

    # If any objects are detected
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]   # Get the class name from class ID

            # If the detected object's name matches the list
            if className in objects:
                objectInfo.append([box, className])
                print(f"Detected {className} with confidence {round(confidence * 100, 2)}%")

                # If drawing is enabled, draw rectangle and labels
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)  # Draw bounding box
                    cv2.putText(img,className.upper(),(box[0]+10,box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)  # Display class name
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                             cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)  # Display confidence level

    return img,objectInfo  # Display confidence level

#try connecting to arduino outside the main loop
arduino = None
try:
    #Use COM port instead of /dev/ttyACM0 for windows
    arduino = serial.Serial('COM3',9600)  # Adjust COM3 to your actual port
    print ("Successfully connected to Arduino")
    time.sleep(2) #wait for arduino to reset
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    print ("Continuing without Arduino connection....")


# Main function to capture video and perform live object detection
if __name__ == '__main__':
    # Initialize webcam capture
    camera_indices = [0,1,2]
    cap = None

    for idx in camera_indices:
        print(f"Trying to open camera with index {idx}....")
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Successfully opened camera with index {idx}")
            break
        else:
            print(f"Failed to open camera with index {idx}")

    if cap is None or not cap.isOpened():
        print("Failed to open camera")
        exit()

    # Set the resolution of the video capture
    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height

    print("starting video capture loop...")

    #List of objects to detect - expanding beyond just elephants
    objects_to_detect = ['person','chair','bottle','laptop','elephant']
    print(f"Looking for objects: {objects_to_detect}")


    # Infinite loop to continuously capture frames
    frame_count = 0
    try:
        while True:
            success, img = cap.read()  # Read a frame from the webcam

            if not success:
                print("Failed to capture frame")
                break

            frame_count += 1
            if frame_count % 30 == 0: #print every 30 frames to avoid console spam
                print(f"Successfully captured frame {frame_count}")

            # Run object detection on the captured frame
            results, objectInfo = getObjects(img, objects=objects_to_detect)

            #if objects are detected and Arduino is connected, send data
            if arduino and objectInfo:
                try:
                    arduino.write(b'1') #send byte '1' when object is detected
                    print ("Sent '1' to Arduino (object detected).")

                    #wait for arduino to reply with a timeout
                    start_time = time.time()
                    while arduino.in_waiting == 0:
                        # Timeout after 0.5 seconds
                        if time.time() - start_time > 0.5:
                            print("No response from Arduino (timeout).")
                            break
                        time.sleep(0.01)

                    if arduino.in_waiting > 0:
                        response = arduino.readline().decode().rstrip()
                        print(f"Arduino replied: {response}")
                except Exception as e:
                    print(f"Error communication with arduino: {e}")

            #Display the detection results
            cv2.imshow("Detection", img)

            #Exit loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting loop...")
                break


    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        #Clean up resources
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if arduino is not None:
             arduino.close()
        print("Resources released, program terminated.")



