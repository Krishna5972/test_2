import threading


class Watchdog:
    def __init__(self, timeout=60):
        self.timeout = timeout
        self._t = None

    def do_expire(self):
        raise Exception('Connection timeout. Watchdog triggered.')

    def start(self):
        if self._t is None:
            self._t = threading.Timer(self.timeout, self.do_expire)
            self._t.start()

    def reset(self):
        if self._t is not None:
            self.stop()
            self.start()

    def stop(self):
        if self._t is not None:
            self._t.cancel()
            self._t = None