import cv2
import numpy as np
import serial
import time
import logging
import argparse
import threading
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import os
import sys

# Configure the logging system
logging.basicConfig(
    filename='elephant_detection.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class ElephantDetectionSystem:
    """Main class implementing the elephant detection system."""

    def __init__(self, camera_source=0, serial_port='/dev/ttyACM0', baud_rate=9600,
                 min_contour_area=10000, blur_kernel=(21, 21), history=500):
        """Initialize the elephant detection system."""
        # Store configuration parameters
        self.min_contour_area = min_contour_area
        self.blur_kernel = blur_kernel
        self.camera_source = camera_source
        self.serial_port = serial_port
        self.baud_rate = baud_rate

        # System state tracking variables
        self.last_detection_time = None
        self.detection_active = False
        self.running = False
        self.paused = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.log_queue = queue.Queue(maxsize=100)
        self.detection_count = 0

        # For storing current frame
        self.current_frame = None

        # Event to signal thread termination
        self.stop_event = threading.Event()

        # Initialize camera and Arduino in separate methods so we can retry connections
        self.cap = None
        self.arduino = None
        self.bg_subtractor = None

        logging.info("Elephant Detection System object created")

    def connect_camera(self):
        """Establish connection to the camera."""
        try:
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.camera_source)
            if not self.cap.isOpened():
                raise IOError("Cannot open camera or video source")

            # Get camera details
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            logging.info(f"Camera connected: {width}x{height} at {fps} FPS")
            self.log_message(f"Camera connected: {width:.0f}x{height:.0f} at {fps:.1f} FPS")

            # Initialize background subtractor
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=False
            )
            return True
        except Exception as e:
            logging.error(f"Camera connection error: {e}")
            self.log_message(f"ERROR: Could not connect to camera: {e}")
            return False

    def connect_arduino(self):
        """Establish connection to the Arduino."""
        try:
            if self.arduino is not None:
                self.arduino.close()

            self.arduino = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            time.sleep(2)  # Allow time for connection to establish

            logging.info(f"Arduino connected on {self.serial_port}")
            self.log_message(f"Arduino connected on {self.serial_port}")
            return True
        except serial.SerialException as e:
            self.arduino = None
            logging.warning(f"Failed to connect to Arduino: {e}")
            self.log_message(f"WARNING: Arduino connection failed - running in camera-only mode")
            return False

    def process_frame(self, frame):
        """Process a video frame to detect large objects (elephants)."""
        # Store the current frame for potential saving
        self.current_frame = frame.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        # Perform background subtraction
        fg_mask = self.bg_subtractor.apply(blurred)

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours and check for elephants
        is_elephant_detected = False
        max_contour_area = 0
        largest_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_contour_area:
                max_contour_area = area
                largest_contour = contour

        # Check if contour exceeds threshold for elephant detection
        if max_contour_area > self.min_contour_area:
            is_elephant_detected = True

            # Draw the largest contour
            if largest_contour is not None:
                cv2.drawContours(frame, [largest_contour], 0, (0, 0, 255), 3)

                # Calculate and display contour area
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Area: {max_contour_area:.0f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add detection status to frame
        status_text = "ELEPHANT DETECTED!" if is_elephant_detected else "No Detection"
        status_color = (0, 0, 255) if is_elephant_detected else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # Add timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Add threshold value
        cv2.putText(frame, f"Threshold: {self.min_contour_area}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame, is_elephant_detected

    def send_alert_to_arduino(self):
        """Send alert signal to Arduino to activate the buzzer."""
        if self.arduino:
            try:
                self.arduino.write(b'1')
                logging.info("Alert signal sent to Arduino")
                self.log_message("Alert signal sent to Arduino")
            except serial.SerialException as e:
                logging.error(f"Failed to send alert to Arduino: {e}")
                self.log_message(f"ERROR: Failed to send alert to Arduino: {e}")

    def stop_alert_to_arduino(self):
        """Send stop signal to Arduino to deactivate the buzzer."""
        if self.arduino:
            try:
                self.arduino.write(b'0')
                logging.info("Stop signal sent to Arduino")
                self.log_message("Stop signal sent to Arduino")
            except serial.SerialException as e:
                logging.error(f"Failed to send stop signal to Arduino: {e}")
                self.log_message(f"ERROR: Failed to send stop signal to Arduino: {e}")

    def log_detection_event(self):
        """Log elephant detection event with timestamp."""
        current_time = datetime.now()
        self.last_detection_time = current_time
        self.detection_count += 1
        message = f"ALERT: Elephant detected at {current_time}"
        logging.info(message)
        self.log_message(message)

        # Save detection image if directory exists
        try:
            if not os.path.exists("detections"):
                os.makedirs("detections")

            if self.current_frame is not None:
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                filename = f"detections/elephant_detection_{timestamp}.jpg"
                cv2.imwrite(filename, self.current_frame)
                self.log_message(f"Detection image saved as {filename}")
        except Exception as e:
            logging.error(f"Failed to save detection image: {e}")
            self.log_message(f"ERROR: Failed to save detection image: {e}")

    def log_message(self, message):
        """Add message to the log queue for GUI display."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")

    def camera_loop(self):
        """Main camera processing loop that runs in a separate thread."""
        logging.info("Camera thread started")
        self.log_message("Camera processing started")

        last_fps_time = time.time()
        frame_count = 0
        fps = 0

        while not self.stop_event.is_set():
            if self.paused:
                time.sleep(0.1)
                continue

            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to capture frame")
                self.log_message("ERROR: Failed to capture frame - check camera connection")
                time.sleep(1)  # Don't hammer the system if camera fails
                continue

            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = frame_count / (current_time - last_fps_time)
                frame_count = 0
                last_fps_time = current_time

            # Add FPS to frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Process frame for elephant detection
            processed_frame, elephant_detected = self.process_frame(frame)

            # Handle detection state changes
            if elephant_detected and not self.detection_active:
                self.detection_active = True
                self.log_detection_event()
                self.send_alert_to_arduino()
            elif not elephant_detected and self.detection_active:
                self.detection_active = False
                self.stop_alert_to_arduino()

            # Update the frame for the GUI
            # We use try/except with a maxsize=1 queue to avoid blocking
            try:
                # Convert BGR to RGB for Tkinter
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                self.frame_queue.put(rgb_frame, False)
            except queue.Full:
                # Skip frame if queue is full (GUI is slower than processing)
                pass

            # Short sleep to reduce CPU usage
            time.sleep(0.01)

        # Cleanup when thread exits
        logging.info("Camera thread stopped")

    def start(self):
        """Start the detection system."""
        if self.running:
            self.log_message("System is already running")
            return False

        # Connect to camera and Arduino
        if not self.connect_camera():
            return False

        try:
            self.connect_arduino()  # We allow this to fail and run in camera-only mode

            # Reset state
            self.stop_event.clear()
            self.running = True
            self.paused = False

            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

            logging.info("Elephant Detection System started")
            self.log_message("System started successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to start system: {e}")
            self.log_message(f"ERROR: Failed to start system: {e}")
            self.cleanup()
            return False

    def stop(self):
        """Stop the detection system."""
        if not self.running:
            return

        # Signal threads to stop
        self.stop_event.set()

        # Wait for thread to finish
        if hasattr(self, 'camera_thread'):
            self.camera_thread.join(timeout=1.0)

        self.cleanup()
        self.running = False
        logging.info("System stopped")
        self.log_message("System stopped")

    def toggle_pause(self):
        """Pause or resume the detection system."""
        if not self.running:
            return

        self.paused = not self.paused
        status = "paused" if self.paused else "resumed"
        logging.info(f"System {status}")
        self.log_message(f"System {status}")
        return self.paused

    def cleanup(self):
        """Clean up resources."""
        if self.arduino:
            self.stop_alert_to_arduino()
            self.arduino.close()
            self.arduino = None

        if self.cap:
            self.cap.release()
            self.cap = None

        logging.info("Resources cleaned up")


class ElephantDetectionUI:
    """
    Graphical User Interface for the Elephant Detection System.

    This class handles the Tkinter UI components and interfaces with
    the ElephantDetectionSystem class.
    """

    def __init__(self, root):
        """Initialize the UI with the given root window."""
        self.root = root
        self.root.title("Elephant Detection System")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initialize the detection system
        self.detector = ElephantDetectionSystem()

        # Create UI elements
        self.create_ui()

        # Start update loop
        self.update_ui()

    def create_ui(self):
        """Create all UI elements."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Split into left and right panels
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        # --- Left Panel (Video Feed) ---
        video_frame = ttk.LabelFrame(left_frame, text="Live Detection Feed")
        video_frame.pack(fill=tk.BOTH, expand=True)

        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status bar under video
        status_frame = ttk.Frame(left_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.status_label = ttk.Label(status_frame, text="Status: Not running")
        self.status_label.pack(side=tk.LEFT)

        self.detection_label = ttk.Label(status_frame, text="Detections: 0")
        self.detection_label.pack(side=tk.RIGHT)

        # --- Right Panel (Controls and Logs) ---
        # Controls section
        controls_frame = ttk.LabelFrame(right_frame, text="System Controls")
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        # Camera settings section
        camera_frame = ttk.Frame(controls_frame)
        camera_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(camera_frame, text="Camera Source:").grid(row=0, column=0, sticky=tk.W)
        self.camera_var = tk.StringVar(value="0")
        camera_entry = ttk.Entry(camera_frame, textvariable=self.camera_var, width=10)
        camera_entry.grid(row=0, column=1, padx=5)

        # Serial port settings
        serial_frame = ttk.Frame(controls_frame)
        serial_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(serial_frame, text="Serial Port:").grid(row=0, column=0, sticky=tk.W)

        # Default serial port based on OS
        default_port = '/dev/ttyACM0'  # Linux/Mac default
        if sys.platform.startswith('win'):
            default_port = 'COM3'  # Windows default

        self.serial_var = tk.StringVar(value=default_port)
        serial_entry = ttk.Entry(serial_frame, textvariable=self.serial_var, width=15)
        serial_entry.grid(row=0, column=1, padx=5)

        # Sensitivity settings
        sensitivity_frame = ttk.Frame(controls_frame)
        sensitivity_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(sensitivity_frame, text="Detection Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.threshold_var = tk.IntVar(value=15000)

        threshold_scale = ttk.Scale(sensitivity_frame, from_=1000, to=50000,
                                    variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.grid(row=0, column=1, padx=5, sticky=tk.EW)

        threshold_label = ttk.Label(sensitivity_frame, textvariable=self.threshold_var)
        threshold_label.grid(row=0, column=2, padx=5)

        # Control buttons
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=10)

        self.start_btn = ttk.Button(buttons_frame, text="Start System", command=self.start_system)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = ttk.Button(buttons_frame, text="Stop System", command=self.stop_system, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.pause_btn = ttk.Button(buttons_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_btn.grid(row=0, column=2, padx=5)

        self.save_btn = ttk.Button(buttons_frame, text="Save Frame", command=self.save_frame, state=tk.DISABLED)
        self.save_btn.grid(row=0, column=3, padx=5)

        # Status indicators
        indicator_frame = ttk.Frame(controls_frame)
        indicator_frame.pack(fill=tk.X, padx=5, pady=5)

        self.camera_indicator = ttk.Label(indicator_frame, text="◯ Camera", foreground="gray")
        self.camera_indicator.grid(row=0, column=0, padx=15)

        self.arduino_indicator = ttk.Label(indicator_frame, text="◯ Arduino", foreground="gray")
        self.arduino_indicator.grid(row=0, column=1, padx=15)

        self.detection_indicator = ttk.Label(indicator_frame, text="◯ Detection", foreground="gray")
        self.detection_indicator.grid(row=0, column=2, padx=15)

        # Log section
        log_frame = ttk.LabelFrame(right_frame, text="System Log")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)

        # Add initial log message
        self.add_log("Elephant Detection System initialized")
        self.add_log("Click 'Start System' to begin")

        # Set focus to start button
        self.start_btn.focus_set()

    def start_system(self):
        """Start the detection system with current settings."""
        # Update detector settings from UI
        try:
            camera_source = self.camera_var.get()
            # Try to convert to integer (camera index) if possible
            try:
                camera_source = int(camera_source)
            except ValueError:
                # Keep as string (likely a file path)
                pass

            self.detector.camera_source = camera_source
            self.detector.serial_port = self.serial_var.get()
            self.detector.min_contour_area = self.threshold_var.get()

            if self.detector.start():
                self.update_button_states(running=True)
                self.add_log("System started")
            else:
                self.add_log("Failed to start system")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start system: {e}")
            logging.error(f"UI error when starting system: {e}")

    def stop_system(self):
        """Stop the detection system."""
        try:
            self.detector.stop()
            self.update_button_states(running=False)
            self.update_indicators(camera=False, arduino=False, detection=False)
            self.add_log("System stopped")
        except Exception as e:
            messagebox.showerror("Error", f"Error when stopping system: {e}")
            logging.error(f"UI error when stopping system: {e}")

    def toggle_pause(self):
        """Pause or resume the detection system."""
        try:
            paused = self.detector.toggle_pause()
            self.pause_btn.config(text="Resume" if paused else "Pause")
            status = "paused" if paused else "resumed"
            self.add_log(f"System {status}")
        except Exception as e:
            messagebox.showerror("Error", f"Error when toggling pause: {e}")
            logging.error(f"UI error when toggling pause: {e}")

    def save_frame(self):
        """Save the current frame as an image file."""
        try:
            if self.detector.current_frame is not None:
                if not os.path.exists("captures"):
                    os.makedirs("captures")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captures/frame_{timestamp}.jpg"
                cv2.imwrite(filename, self.detector.current_frame)
                self.add_log(f"Frame saved as {filename}")
            else:
                self.add_log("No frame available to save")
        except Exception as e:
            self.add_log(f"Error saving frame: {e}")
            logging.error(f"Error saving frame: {e}")

    def update_button_states(self, running=False):
        """Update button states based on system state."""
        if running:
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
        else:
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.DISABLED, text="Pause")
            self.save_btn.config(state=tk.DISABLED)

    def update_indicators(self, camera=False, arduino=False, detection=False):
        """Update status indicators."""
        self.camera_indicator.config(
            text="● Camera" if camera else "◯ Camera",
            foreground="green" if camera else "gray"
        )

        self.arduino_indicator.config(
            text="● Arduino" if arduino else "◯ Arduino",
            foreground="green" if arduino else "gray"
        )

        self.detection_indicator.config(
            text="● Detection" if detection else "◯ Detection",
            foreground="red" if detection else "gray"
        )

    def add_log(self, message):
        """Add a message to the log text widget."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_ui(self):
        """Update UI elements with latest data from detection system."""
        # Update video feed
        try:
            if not self.detector.frame_queue.empty():
                frame = self.detector.frame_queue.get(False)
                self.update_video_feed(frame)
        except Exception:
            pass

        # Update logs
        while not self.detector.log_queue.empty():
            try:
                message = self.detector.log_queue.get(False)
                self.add_log(message)
            except Exception:
                break

        # Update status information
        if self.detector.running:
            status = "Paused" if self.detector.paused else "Running"
            self.status_label.config(text=f"Status: {status}")

            # Update detection count
            self.detection_label.config(text=f"Detections: {self.detector.detection_count}")

            # Update threshold value for the detector in case it was changed
            self.detector.min_contour_area = self.threshold_var.get()

            # Update indicators
            self.update_indicators(
                camera=True,
                arduino=self.detector.arduino is not None,
                detection=self.detector.detection_active
            )
        else:
            self.status_label.config(text="Status: Not running")

        # Schedule the next update
        self.root.after(50, self.update_ui)

    def update_video_feed(self, frame):
        """Update the video feed with a new frame."""
        if frame is None:
            return

        # Resize frame to fit canvas while maintaining aspect ratio
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        if canvas_width > 10 and canvas_height > 10:  # Ensure canvas has valid size
            # Calculate scaling factor
            frame_height, frame_width = frame.shape[:2]

            scale_width = canvas_width / frame_width
            scale_height = canvas_height / frame_height
            scale = min(scale_width, scale_height)

            # Resize frame
            if scale < 1:  # Only resize if frame is larger than canvas
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)
                frame = cv2.resize(frame, (new_width, new_height),
                                   interpolation=cv2.INTER_AREA)

        # Convert to PhotoImage format for Tkinter
        pil_img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=pil_img)

        # Update canvas
        self.video_canvas.config(width=pil_img.width, height=pil_img.height)
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.video_canvas.image = img_tk  # Keep reference to prevent garbage collection

    def on_close(self):
        """Handle window close event."""
        if self.detector.running:
            if messagebox.askokcancel("Quit", "System is running. Stop and quit?"):
                self.stop_system()
                self.root.destroy()
        else:
            self.root.destroy()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Elephant Detection System with GUI')
    parser.add_argument('--no-gui', action='store_true', help='Run in headless mode (no GUI)')

    args = parser.parse_args()

    if args.no_gui:
        # Run in headless mode (command line only)
        try:
            print("Starting Elephant Detection System in headless mode...")
            detector = ElephantDetectionSystem()
            if detector.start():
                print("System started. Press Ctrl+C to stop.")
                while detector.running:
                    time.sleep(1)
            else:
                print("Failed to start system.")
        except KeyboardInterrupt:
            print("\nShutting down...")
            detector.stop()
            print("System stopped.")
    else:
        # Run with GUI
        root = tk.Tk()
        app = ElephantDetectionUI(root)
        root.mainloop()