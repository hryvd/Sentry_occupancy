"""
scripts/launch_ui.py
Launch local SENTRY Parking Occupancy Cognitive Visualizer in the default browser.
Usage:
    python scripts/launch_ui.py
    python scripts/launch_ui.py --port 8088
"""

import argparse
import http.server
import socketserver
import webbrowser
from pathlib import Path

DEFAULT_PORT = 8080


def main():
    parser = argparse.ArgumentParser(description="Launch local SENTRY parking occupancy visualizer.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to serve on")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    if not (ui_dir / "index.html").exists():
        raise FileNotFoundError(f"Visualizer index.html not found in {ui_dir}")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ui_dir), **kwargs)

    url = f"http://localhost:{args.port}/index.html"
    print(f"\n========================================================")
    print(f" SENTRY Parking Occupancy Cognitive Visualizer UI")
    print(f" Serving UI from: {ui_dir}")
    print(f" URL: {url}")
    print(f" Press Ctrl+C to stop.")
    print(f"========================================================\n")

    if not args.no_browser:
        webbrowser.open(url)

    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down visualizer server.")


if __name__ == "__main__":
    main()
