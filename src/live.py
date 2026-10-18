"""Loopback-only dashboard for observing the first actual PPO training lane."""
import base64
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time
import webbrowser
from PIL import Image
from .sim.render import Renderer


class LiveTrainingView:
    def __init__(self, *, camera='overview', fps=10, port=0, open_browser=True):
        self.renderer = Renderer(1, camera=camera)
        self.camera = camera
        self.period = 1 / fps
        self.last_frame = None
        self.lock = threading.Lock()
        self.state = {'phase': 'Starting', 'frame': None, 'camera': camera, 'metrics': None}
        html = Path(__file__).with_name('live.html').read_bytes()
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send(self, status, content, content_type):
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                try:
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path == '/':
                    self.send(200, html, 'text/html; charset=utf-8')
                elif self.path == '/state':
                    with view.lock:
                        data = json.dumps(view.state, allow_nan=False).encode()
                    self.send(200, data, 'application/json')
                else:
                    self.send(404, b'Not found', 'text/plain')

            def do_POST(self):
                origin = self.headers.get('Origin')
                if self.path != '/camera' or origin not in (None, view.url):
                    self.send(403, b'Forbidden', 'text/plain'); return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 128 or self.headers.get('Content-Type') != 'application/json':
                        raise ValueError('Invalid request')
                    camera = json.loads(self.rfile.read(length))['camera']
                    if camera not in ('follow', 'overview'):
                        raise ValueError('Invalid camera')
                except (ValueError, KeyError, TypeError):
                    self.send(400, b'Invalid camera request', 'text/plain'); return
                with view.lock:
                    view.camera = camera
                self.send(200, b'{}', 'application/json')

        try:
            self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        except Exception:
            self.renderer.close()
            raise
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f'Live training: {self.url}', flush=True)
        if open_browser:
            webbrowser.open(self.url)

    def status(self, **values):
        with self.lock:
            self.state.update(values)

    def frame(self, env, *, pace=True, **values):
        if pace and self.last_frame is not None:
            time.sleep(max(0, self.period - (time.monotonic() - self.last_frame)))
        self.last_frame = time.monotonic()
        with self.lock:
            camera = self.camera
        if self.renderer.camera != camera:
            self.renderer.close()
            self.renderer = Renderer(1, camera=camera)
        # A read-only shallow view: avoid copying/sensing every training car for display.
        lane = copy.copy(env)
        lane.n = 1
        for name in ('pos', 'heading', 'speed', 'last_action', 'obstacles', 'steps'):
            setattr(lane, name, getattr(env, name)[:1])
        rgb = self.renderer.frame(lane, 'live PPO training')
        output = io.BytesIO()
        Image.fromarray(rgb).save(output, format='PNG')
        self.status(frame=base64.b64encode(output.getvalue()).decode(), camera=camera, **values)

    def wait(self, env):
        print('Training complete. Live view stays open; press Ctrl+C in Terminal to exit.', flush=True)
        while True:
            if self.renderer.camera != self.camera:
                self.frame(env, pace=False)
            time.sleep(.2)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.renderer.close()
