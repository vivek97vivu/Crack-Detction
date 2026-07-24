import queue
import time
import threading
import logging

logger = logging.getLogger(__name__)

class InferenceScheduler:
    """
    Manages a thread-safe queue and dynamically batches frames from multiple camera
    threads into GPU inference passes, maximizing TensorRT throughput across worker threads.
    """
    def __init__(self, detector, batch_size=4, timeout_ms=5.0, num_workers=3):
        self.detector = detector
        self.batch_size = batch_size
        self.timeout = timeout_ms / 1000.0  # Convert to seconds
        self.queue = queue.Queue(maxsize=200)
        self.stop_event = threading.Event()
        self.num_workers = num_workers
        self.threads = [
            threading.Thread(target=self._loop, name=f"InferenceScheduler-{i}", daemon=True)
            for i in range(num_workers)
        ]

    def start(self):
        """Starts the scheduler background loop threads."""
        logger.info("Starting %d InferenceScheduler worker threads with batch_size=%d timeout_ms=%.1f",
                    self.num_workers, self.batch_size, self.timeout * 1000.0)
        for t in self.threads:
            t.start()

    def stop(self):
        """Stops the scheduler loop threads."""
        logger.info("Stopping InferenceScheduler threads...")
        self.stop_event.set()
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2.0)

    def submit(self, image):
        """
        Submit an image for inference and block until results are ready.
        Args:
            image (np.ndarray): BGR frame to run prediction on.
        Returns:
            list[dict]: List of raw predictions.
        """
        if self.stop_event.is_set():
            return []

        res_queue = queue.Queue(maxsize=1)
        try:
            self.queue.put((image, res_queue), timeout=1.0)
            return res_queue.get(timeout=5.0)  # Block until result is dispatched
        except queue.Full:
            logger.warning("[Scheduler] Submission queue full, dropping frame.")
            return []
        except queue.Empty:
            logger.warning("[Scheduler] Inference timeout, returning empty detections.")
            return []
        except Exception as e:
            logger.error("[Scheduler] Submission exception: %s", e)
            return []

    def _loop(self):
        """Main loop that collects, batches, executes, and dispatches inference."""
        while not self.stop_event.is_set():
            batch = []
            res_queues = []

            # 1. Wait for the first item to arrive (blocking to save CPU)
            try:
                item = self.queue.get(timeout=0.1)
                batch.append(item[0])
                res_queues.append(item[1])
            except queue.Empty:
                continue

            # 2. Collect up to batch_size items or until timeout occurs
            t0 = time.time()
            while len(batch) < self.batch_size:
                elapsed = time.time() - t0
                if elapsed >= self.timeout:
                    break

                try:
                    rem = self.timeout - elapsed
                    item = self.queue.get(timeout=max(0.0001, rem))
                    batch.append(item[0])
                    res_queues.append(item[1])
                except queue.Empty:
                    break

            # 3. Execute batched prediction on GPU
            if batch:
                try:
                    batch_outputs = self.detector.predict_batch(batch)
                except Exception as exc:
                    logger.error("[Scheduler] Batched prediction failure: %s", exc)
                    batch_outputs = [[] for _ in batch]

                # 4. Dispatch outputs to respective response queues
                for out, rq in zip(batch_outputs, res_queues):
                    try:
                        rq.put(out, block=False)
                    except queue.Full:
                        pass
