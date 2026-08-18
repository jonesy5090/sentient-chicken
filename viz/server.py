"""Static file + tiny JSON/binary API server for the offline trajectory viewer.

Deliberately stdlib-only (`http.server`): the container doesn't need a fresh
`pip install` to serve a page, `python -m viz.server` just runs against the
project's existing venv. Binds 0.0.0.0 so the port can be forwarded to the host
browser. It never talks to the simulation -- only to the flat binary files
`run.record` writes under `runs/`.

    usage:  python -m viz.server --port 8000
"""

import argparse
import http.server
import json
import re
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
WEB_DIR = Path(__file__).resolve().parent / "web"

# Run ids come from directory names `run.record` itself creates
# (`{timestamp}_{slug}`), but the id also arrives back over the network as a URL
# path segment, so it gets the same validation any untrusted path input would.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _run_dir(run_id: str) -> Path | None:
    if not _RUN_ID_RE.match(run_id):
        return None
    d = RUNS_DIR / run_id
    try:
        d.resolve().relative_to(RUNS_DIR.resolve())
    except ValueError:
        return None
    return d if d.is_dir() else None


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, fmt, *args):
        pass  # keep stdout for the startup line, not per-request noise

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/runs":
            return self._list_runs()
        m = re.match(r"^/api/runs/([^/]+)/(meta\.json|trajectory\.bin)$", parsed.path)
        if m:
            return self._run_file(m.group(1), m.group(2))
        return super().do_GET()

    def _list_runs(self):
        runs = []
        if RUNS_DIR.is_dir():
            for d in sorted(RUNS_DIR.iterdir(), reverse=True):
                mf = d / "meta.json"
                if not mf.is_file():
                    continue
                try:
                    meta = json.loads(mf.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                meta.pop("layout", None)  # picker list doesn't need byte offsets
                runs.append(meta)
        self._send_json(runs)

    def _run_file(self, run_id: str, filename: str):
        d = _run_dir(run_id)
        if d is None:
            self.send_error(404, "unknown run")
            return
        f = d / filename
        if not f.is_file():
            self.send_error(404, "missing file")
            return
        body = f.read_bytes()
        self.send_response(200)
        content_type = "application/json" if filename.endswith(".json") \
            else "application/octet-stream"
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0",
                    help="0.0.0.0 so the port can be forwarded to the host")
    args = ap.parse_args()
    RUNS_DIR.mkdir(exist_ok=True)
    httpd = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {WEB_DIR.relative_to(ROOT)} + {len(list(RUNS_DIR.glob('*/meta.json')))} "
          f"recorded run(s) on http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
