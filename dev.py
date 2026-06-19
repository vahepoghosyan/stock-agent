#!/usr/bin/env python3
"""Dev watcher — restarts main.py --run-now on any .py or .json file change."""

import subprocess
import sys
import time
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCH_EXTS = {".py", ".json"}
CMD = [sys.executable, "main.py", "--run-now"]


class Restarter(FileSystemEventHandler):
    def __init__(self):
        self.proc = self._start()

    def _start(self):
        print(">> Starting process...")
        return subprocess.Popen(CMD)

    def on_modified(self, event):
        if Path(event.src_path).suffix in WATCH_EXTS:
            print(f">> Change detected: {event.src_path} — restarting...")
            self.proc.terminate()
            self.proc.wait()
            self.proc = self._start()


handler = Restarter()
observer = Observer()
observer.schedule(handler, path=".", recursive=False)
observer.start()
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
    handler.proc.terminate()
observer.join()
