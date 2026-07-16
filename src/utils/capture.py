import threading
import time
import cv2

class ThreadedVideoCapture:
    """
    A wrapper around cv2.VideoCapture that performs frame grabbing in a 
    background thread. This prevents FFMPEG buffer accumulation, reduces CPU 
    overhead, and ensures that the main thread always retrieves the most 
    recent frame.
    """
    def __init__(self, source, api_preference=cv2.CAP_ANY):
        if isinstance(source, cv2.VideoCapture):
            self.cap = source
        else:
            self.cap = cv2.VideoCapture(source, api_preference)
            
        self.grabbed = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        
        # Read the first frame synchronously to initialize the buffer
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            self.grabbed = ret
            self.frame = frame
            self.new_frame = True
            
            # Start background frame grabber thread
            self.thread = threading.Thread(target=self._update, args=())
            self.thread.daemon = True
            self.thread.start()
            
    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
                
            # cv2.VideoCapture.grab() is lightweight and clears the buffer
            grabbed = self.cap.grab()
            if not grabbed:
                with self.lock:
                    self.grabbed = False
                time.sleep(0.01)
                continue
                
            # cv2.VideoCapture.retrieve() fetches the grabbed frame
            ret, frame = self.cap.retrieve()
            with self.lock:
                self.grabbed = ret
                if ret:
                    self.frame = frame
                    self.new_frame = True
            time.sleep(0.001)  # brief yield to prevent thread hogging
            
    def read(self):
        """Returns the latest grabbed frame. Blocks if no new frame is available."""
        # Wait until a new frame is grabbed by the background thread
        start_time = time.time()
        while not self.stopped and not self.new_frame:
            time.sleep(0.001)
            # Timeout after 1.0 second to prevent permanent blocking
            if time.time() - start_time > 1.0:
                break
                
        with self.lock:
            self.new_frame = False
            grabbed = self.grabbed
            # Copy frame to prevent race conditions during rendering/inference
            frame = self.frame.copy() if (self.frame is not None) else None
        return grabbed, frame
        
    def isOpened(self) -> bool:
        return self.cap.isOpened()
        
    def release(self):
        """Stops the thread and releases the Capture object."""
        self.stopped = True
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        self.cap.release()
        
    def get(self, propId):
        return self.cap.get(propId)
        
    def set(self, propId, value):
        return self.cap.set(propId, value)
        
    def getBackendName(self) -> str:
        return self.cap.getBackendName()
