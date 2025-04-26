# Importing necessary libraries
import cv2
import numpy as np

# Setting the detection threshold
thres = 0.45   # Threshold to detect object (confidence level)


# List to hold the class names from the COCO dataset
classNames= []

# Path to the file containing class labels (COCO dataset)
classFile = 'coco.names'

# Reading the class names from the file
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Paths to the model configuration and pre-trained weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'


# Loading the pre-trained deep learning model using OpenCV's DNN module
net = cv2.dnn_DetectionModel(weightsPath,configPath)

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
                objectInfo.append([box])

                # If drawing is enabled, draw rectangle and labels
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)  # Draw bounding box
                    cv2.putText(img,className.upper(),(box[0]+10,box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)  # Display class name
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                             cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)  # Display confidence level

    return img,objectInfo  # Display confidence level


# Main function to capture video and perform live object detection
if __name__ == '__main__':
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)

    # Set the resolution of the video capture
    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height

    # Infinite loop to continuously capture frames
    while True:
        success, img = cap.read()  # Read a frame from the webcam

        # Run object detection on the captured frame
        results, objectInfo = getObjects(img, objects=['elephant'])  # Only look for 'elephant'

        # Display the detection results
        cv2.imshow("Detection", img)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


