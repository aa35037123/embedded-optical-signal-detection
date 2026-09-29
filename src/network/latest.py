"""Continuously drain TCP into one replaceable packet slot."""
import socket
import threading
import time

from .protocol import ConnectionClosed, FrameSequence, receive_packet


class LatestPacketReceiver:
    def __init__(self, connection, metrics):
        self.connection = connection
        self.metrics = metrics
        self.condition = threading.Condition()
        self.pending = None
        self.done = False
        self.error = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True, name='tcp-frame-reader')

    def start(self):
        self.thread.start()
        return self

    def _read(self):
        sequence = FrameSequence()
        try:
            while not self.stopping.is_set():
                previous_skips = sequence.skipped_frames
                packet = receive_packet(self.connection, sequence)
                timestamp = time.monotonic_ns()
                self.metrics.received(packet, timestamp, sequence.skipped_frames - previous_skips)
                self.publish(packet, timestamp)
        except ConnectionClosed as exc:
            if not (exc.phase == 'header' and exc.received == 0) and not self.stopping.is_set():
                self.error = exc
        except Exception as exc:
            if not self.stopping.is_set():
                self.error = exc
        finally:
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def publish(self, packet, timestamp):
        with self.condition:
            if self.pending is not None:
                self.metrics.overwritten()
            self.pending = (packet, timestamp)
            self.condition.notify()

    def take(self, timeout=0.1):
        with self.condition:
            self.condition.wait_for(lambda: self.pending is not None or self.done, timeout)
            if self.error is not None:
                raise self.error
            value, self.pending = self.pending, None
            return value

    def stop(self):
        self.stopping.set()
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            raise RuntimeError('Network reader did not stop')
