"""Serve the standard Swift page with compact, live jog controls."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlsplit
import importlib

import swift


class PendantServer:
    def __init__(self, outq, inq, socket_port, run):
        root = Path(swift.__file__).parent / "out"
        css = Path(__file__).with_name("teach_pendant.css").read_text()
        script = Path(__file__).with_name("teach_pendant.js").read_text()
        page = (root / "index.html").read_text(encoding="utf-8")
        page = page.replace("</head>", f"<style>{css}</style></head>")
        page = page.replace("</body>", f"<script>{script}</script></body>").encode("utf-8")

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(root), **kwargs)

            def send_head(self):
                if urlsplit(self.path).path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    return BytesIO(page)
                return super().send_head()

            def translate_path(self, path):
                # keep the normal swift mesh retrieval route
                if path.startswith("/retrieve/"):
                    return unquote(urlsplit(path).path[len("/retrieve/"):])
                return super().translate_path(path)

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("localhost", 0), Handler) as server:
            server.timeout = 0.2
            inq.put(server.server_port)
            while run():
                server.handle_request()


def launch_swift(simulator):
    # use a local page override without changing the installed swift package
    route = importlib.import_module("swift.SwiftRoute")
    original_server = route.SwiftServer
    route.SwiftServer = PendantServer
    try:
        simulator.launch(realtime=True)
    finally:
        route.SwiftServer = original_server
