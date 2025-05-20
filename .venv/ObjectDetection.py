# Elephant Detection System with Alarm
# ------------------------------------------------------------------------------
# This script implements a computer vision system that detects elephants (and other
# objects) using a pre-trained neural network model. When an elephant is detected,
# the system activates an alarm (LED and buzzer) connected to a Raspberry Pi's GPIO pins.
# ------------------------------------------------------------------------------

# Import necessary libraries
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy for numerical operations
import time  # Time library for timing functions and delays
import RPi.GPIO as GPIO  # Library to control Raspberry Pi GPIO pins

# Define the confidence threshold for object detection
# Any detection with confidence below this value will be ignored
thres = 0.45  # 45% confidence threshold for object detection

# Define GPIO pin assignments for the alarm components
BUZZER_PIN = 2  # GPIO pin connected to the buzzer (BCM numbering)
LED_PIN = 27  # GPIO pin connected to the LED indicator (BCM numbering)


# Function to initialize and configure the GPIO pins
def setup_gpio():
    """
    Set up the Raspberry Pi GPIO pins for the alarm system.
    Returns True if successful, False otherwise.
    """
    try:
        # Configure GPIO settings
        GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering scheme
        GPIO.setwarnings(False)  # Disable warnings for already configured pins

        # Set up pins as outputs
        GPIO.setup(BUZZER_PIN, GPIO.OUT)  # Buzzer pin as output
        GPIO.setup(LED_PIN, GPIO.OUT)  # LED pin as output

        # Set initial states (everything off)
        # Note: For active buzzers, HIGH typically means OFF
        GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Buzzer off
        GPIO.output(LED_PIN, GPIO.LOW)  # LED off

        # Note: The following code is commented out but would be used for passive buzzers
        # Create PWM object for passive buzzer
        # buzzer_pwm = GPIO.PWM(BUZZER_PIN, 1000)  # 1000 Hz frequency
        # buzzer_pwm.start(0)  # Start with 0% duty cycle (off)

        print("GPIO initialized successfully")

        # Run a quick visual indicator sequence to show system is starting
        for _ in range(3):  # Blink the LED 3 times
            GPIO.output(LED_PIN, GPIO.HIGH)  # LED on
            time.sleep(0.1)  # Wait 100ms
            GPIO.output(LED_PIN, GPIO.LOW)  # LED off
            time.sleep(0.1)  # Wait 100ms

        # Quick buzzer test to confirm it's working
        GPIO.output(BUZZER_PIN, GPIO.LOW)  # Turn buzzer on briefly
        time.sleep(0.2)  # Sound for 200ms
        GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Turn buzzer off

        return True  # GPIO setup successful

    except Exception as e:
        # If anything goes wrong during setup, report the error
        print(f"Error setting up GPIO: {e}")
        return False  # GPIO setup failed


# Function to control the alarm system
def control_alarm(state):
    """
    Control the alarm components (LED and buzzer) based on detection state.

    Parameters:
    state (bool): True to activate alarm, False to deactivate

    Returns:
    bool: True if operation was successful
    """
    if state:  # If activating the alarm
        print("ALERT: Elephant detected!")
        GPIO.output(LED_PIN, GPIO.HIGH)  # Turn on LED
        # Note: Buzzer control is handled in the main loop for pulsing effect
        return True
    else:  # If deactivating the alarm
        print("Alert cleared")
        GPIO.output(LED_PIN, GPIO.LOW)  # Turn off LED
        GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Turn off buzzer
        return True


# Initialize empty list to hold class names from the COCO dataset
# These are the object categories the model can detect
classNames = []

# Define the path to the file containing the class names
classFile = 'coco.names'  # Standard COCO dataset labels file

# Try to read the class names from the file
try:
    with open(classFile, 'rt') as f:
        # Read the file, removing trailing newlines and splitting by line
        classNames = f.read().rstrip('\n').split('\n')
    print(f"Successfully loaded {len(classNames)} class names from {classFile}")
except Exception as e:
    # If the file cannot be loaded, report the error and exit
    print(f"Error loading class names: {e}")
    exit()

# Define paths to the model files
# This uses a pre-trained SSD (Single Shot MultiBox Detector) with MobileNet backbone
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Model configuration
weightsPath = 'frozen_inference_graph.pb'  # Model weights

# Load the pre-trained deep learning model using OpenCV's DNN module
try:
    # Create a detection model instance with the specified weights and config
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    print("Successfully loaded DNN model")
except Exception as e:
    # If the model cannot be loaded, report the error and exit
    print(f"Error loading the model: {e}")
    exit()

# Configure the neural network input parameters
# These settings are specific to the SSD MobileNet model being used
net.setInputSize(320, 320)  # Set input size to 320x320 pixels (model requirement)
net.setInputScale(1.0 / 127.5)  # Scale input pixel values (normalize to [-1,1])
net.setInputMean((127.5, 127.5, 127.5))  # Subtract mean values from each channel
net.setInputSwapRB(True)  # Swap Red and Blue channels (OpenCV uses BGR, model expects RGB)


# Function to detect objects in an image
def getObjects(img, draw=True, objects=[]):
    """
    Detects specified objects in the input image using the pre-trained model.

    Parameters:
    img (numpy.ndarray): The input image/frame to process
    draw (bool): Whether to draw bounding boxes and labels on the image
    objects (list): List of specific object classes to detect (empty for all)

    Returns:
    tuple: (
        img: Annotated image with bounding boxes if draw=True
        objectInfo: List of detected objects with their bounding boxes
        detectedClasses: List of class names that were detected
    )
    """

    # Run object detection on the image
    # Returns class IDs, confidence scores, and bounding boxes
    # nmsThreshold controls non-maximum suppression (removes overlapping boxes)
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=0.2)

    # If no specific objects are specified, detect all known classes
    if len(objects) == 0:
        objects = classNames

    objectInfo = []  # List to store information about detected objects
    detectedClasses = []  # List to store names of detected classes

    # Process detection results if any objects were detected
    if len(classIds) != 0:
        # Loop through each detected object
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Get the class name from the class ID (IDs start at 1, so subtract 1 for list index)
            className = classNames[classId - 1]

            # Check if the detected class is in our target list
            if className in objects:
                # Store the object's information (bounding box and class name)
                objectInfo.append([box, className])
                # Add the class name to the list of detected classes
                detectedClasses.append(className)
                # Print detection details
                print(f"Detected {className} with confidence {round(confidence * 100, 2)}%")

                # If drawing is enabled, visualize the detection
                if draw:
                    # Draw a green rectangle around the object
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

                    # Display the class name above the bounding box
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    # Display the confidence percentage
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Return the processed image and detection information
    return img, objectInfo, detectedClasses


# Main function - program entry point
if __name__ == '__main__':
    # Initialize GPIO pins for the alarm system
    gpio_ready = setup_gpio()
    if not gpio_ready:
        print("WARNING: GPIO initialization failed, continuing without alarm functionality")

    # Initialize the camera/webcam
    # Try several possible camera indices (0, 1, 2) to find the correct one
    camera_indices = [0, 1, 2]
    cap = None  # VideoCapture object

    # Try each camera index until one works
    for idx in camera_indices:
        print(f"Trying to open camera with index {idx}....")
        cap = cv2.VideoCapture(idx)  # Attempt to open camera
        if cap.isOpened():
            print(f"Successfully opened camera with index {idx}")
            break  # Found a working camera, exit the loop
        else:
            print(f"Failed to open camera with index {idx}")

    # If no camera could be opened, exit the program
    if cap is None or not cap.isOpened():
        print("Failed to open camera")
        GPIO.cleanup()  # Clean up GPIO pins before exiting
        exit()

    # Configure camera resolution
    cap.set(3, 1280)  # Width: 1280 pixels
    cap.set(4, 720)  # Height: 720 pixels (720p)

    print("Starting video capture loop...")

    # Define list of objects to detect
    # The model can detect many objects, but we're focusing on these
    objects_to_detect = ['person', 'chair', 'bottle', 'laptop', 'elephant']

    # Define priority objects that should trigger the alarm
    # Currently only elephants will trigger the alarm
    priority_objects = ['elephant']

    print(f"Looking for objects: {objects_to_detect}")
    print(f"Priority objects that trigger alarm: {priority_objects}")

    # Initialize variables for alarm state management
    alarm_active = False  # Is the alarm currently active?
    last_elephant_time = 0  # When was the last elephant detected?
    elephant_timeout = 5  # How long to keep alarm on after detection stops (seconds)

    # Variables for non-blocking buzzer pattern (pulsing effect)
    last_buzzer_toggle = 0  # When was the buzzer last toggled?
    buzzer_state = False  # Is the buzzer currently on?
    buzzer_interval = 0.4  # Toggle interval in seconds (controls pulse rate)

    # Main processing loop
    frame_count = 0  # Counter for processed frames
    try:
        # Run indefinitely until interrupted
        while True:
            # Capture a frame from the camera
            success, img = cap.read()

            # If frame capture failed, exit the loop
            if not success:
                print("Failed to capture frame")
                break

            # Increment frame counter
            frame_count += 1

            # Print status message every 30 frames to avoid console spam
            if frame_count % 30 == 0:
                print(f"Successfully captured frame {frame_count}")

            # Process the captured frame to detect objects
            results, objectInfo, detectedClasses = getObjects(img, objects=objects_to_detect)

            # Check if any priority objects (elephants) were detected
            priority_detected = any(obj in priority_objects for obj in detectedClasses)

            # Display the list of detected objects on the image
            objText = f"Objects: {', '.join(detectedClasses) if detectedClasses else 'None'}"
            cv2.putText(img, objText, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Get current time for alarm management
            current_time = time.time()

            # Handle alarm activation when priority object is detected
            if priority_detected:
                # Update the last detection time
                last_elephant_time = current_time

                # If alarm is not already active, activate it
                if not alarm_active and gpio_ready:
                    if control_alarm(True):
                        alarm_active = True
                        # Display alarm status on the image
                        cv2.putText(img, "ALARM ACTIVE!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 3)

            # Handle alarm deactivation after timeout period
            elif alarm_active and current_time - last_elephant_time > elephant_timeout:
                # Deactivate alarm after timeout period
                if gpio_ready:
                    if control_alarm(False):
                        alarm_active = False

            # Create pulsing buzzer pattern when alarm is active
            if alarm_active and gpio_ready:
                # Check if it's time to toggle the buzzer state
                if current_time - last_buzzer_toggle >= buzzer_interval:
                    # Toggle the buzzer state (on->off or off->on)
                    buzzer_state = not buzzer_state
                    # For active buzzers: LOW is ON, HIGH is OFF
                    GPIO.output(BUZZER_PIN, buzzer_state)
                    # Update the last toggle time
                    last_buzzer_toggle = current_time

            # Display alarm status on the image if active
            if alarm_active:
                cv2.putText(img, "ALARM ACTIVE!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

            # Display the processed image with detection results
            cv2.imshow("Elephant Detection System", img)

            # Check for key press - exit if 'q' is pressed
            # waitKey(1) waits 1ms between frames, needed for UI updates
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting loop...")
                # Turn off alarm before exiting
                if alarm_active and gpio_ready:
                    control_alarm(False)
                break

    # Handle unexpected errors
    except Exception as e:
        print(f"Error in main loop: {e}")

    # Cleanup code that runs regardless of how the loop exits
    finally:
        # Release the camera resource
        if cap is not None:
            cap.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()

        # Clean up GPIO pins to prevent issues with future programs
        if gpio_ready:
            # Ensure alarm is off
            if alarm_active:
                control_alarm(False)
            # Release GPIO resources
            GPIO.cleanup()

        print("Resources released, program terminated.")