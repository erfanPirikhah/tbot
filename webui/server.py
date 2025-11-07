import os
import time
import json
import threading
from datetime import datetime
from typing import List, Dict, Optional

from flask import Flask, jsonify, request, Response, send_file
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix


def _workspace_root() -> str:
    # server.py is placed under webui/, workspace root is parent directory
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _candidate_dirs() -> List[str]:
    root = _workspace_root()
    candidates = [
        os.path.join(root, "logs", "live"),
        os.path.join(root, "v4", "logs", "live"),
        os.path.join(root, "logs"),
        os.path.join(root, "v4", "logs"),
    ]
    return [d for d in candidates if os.path.isdir(d)]


def _gather_files(prefer_jsonl: bool = True) -> List[Dict]:
    found: List[Dict] = []
    for base in _candidate_dirs():
        for dirpath, _, filenames in os.walk(base):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in [".jsonl", ".log", ".txt"]:
                    full = os.path.join(dirpath, name)
                    try:
                        st = os.stat(full)
                        found.append({
                            "name": name,
                            "path": full,
                            "size": st.st_size,
                            "mtime": st.st_mtime,
                            "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(),
                            "kind": ("jsonl" if ext == ".jsonl" else "text"),
                            "dir": dirpath,
                        })
                    except Exception:
                        continue
    # sort newer first; if prefer_jsonl, put jsonl ahead for same mtime
    found.sort(key=lambda f: (f["mtime"], 1 if f["kind"] == "text" and prefer_jsonl else 0), reverse=True)
    return found


def _is_allowed_path(p: str) -> bool:
    try:
        ap = os.path.abspath(p)
        for base in _candidate_dirs():
            if ap.startswith(os.path.abspath(base) + os.sep) or ap == os.path.abspath(base):
                return True
        return False
    except Exception:
        return False


def _tail_sse(file_path: str, from_start: bool = False, as_json: bool = True, heartbeat_seconds: float = 15.0):
    """
    Generator that streams appended lines from file_path as SSE.
    - If as_json is True, tries to parse each line as JSON (for .jsonl).
    - Otherwise sends raw line string.
    - Sends heartbeat comments periodically to keep connection alive.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            if not from_start:
                # seek to end
                f.seek(0, os.SEEK_END)
            last_heartbeat = time.time()

            while True:
                line = f.readline()
                if line:
                    line = line.rstrip("\r\n")
                    payload = None
                    if as_json:
                        try:
                            obj = json.loads(line)
                            payload = json.dumps(obj, ensure_ascii=False, default=str)
                        except Exception:
                            # fallback to raw string
                            payload = json.dumps({"message": line}, ensure_ascii=False)
                    else:
                        payload = json.dumps({"message": line}, ensure_ascii=False)

                    yield f"data: {payload}\n\n"
                    last_heartbeat = time.time()
                else:
                    # no new line, sleep briefly
                    time.sleep(0.5)
                    # heartbeat to keep proxies from closing idle stream
                    if (time.time() - last_heartbeat) >= heartbeat_seconds:
                        yield ": ping\n\n"
                        last_heartbeat = time.time()
    except FileNotFoundError:
        yield f"event: error\ndata: {json.dumps({'error': 'file_not_found'})}\n\n"
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


def create_app() -> Flask:
    app = Flask(__name__, static_folder=".", static_url_path="")
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.route("/", methods=["GET"])
    def index():
        # Serve the local index.html from webui/
        return send_file(os.path.join(os.path.dirname(__file__), "index.html"))

    @app.route("/api/live-files", methods=["GET"])
    def list_live_files():
        files = _gather_files(prefer_jsonl=True)
        return jsonify({
            "count": len(files),
            "files": files,
            "roots": _candidate_dirs(),
        })

    @app.route("/api/last-live", methods=["GET"])
    def last_live():
        files = _gather_files(prefer_jsonl=True)
        # pick newest jsonl if any, otherwise newest text
        if not files:
            return jsonify({"error": "no_files"}), 404
        # ensure first is the most recent
        return jsonify(files[0])

    @app.route("/api/tail", methods=["GET"])
    def tail():
        path = request.args.get("path", "") or ""
        from_start = request.args.get("from_start", "false").lower() in ("1", "true", "yes")
        # decide if JSON parser should be used
        as_json = os.path.splitext(path)[1].lower() == ".jsonl"

        if not path:
            return jsonify({"error": "missing_path"}), 400

        if not _is_allowed_path(path):
            return jsonify({"error": "forbidden_path"}), 403

        # Immediately send a meta event with file info
        try:
            st = os.stat(path)
            meta = {
                "path": path,
                "size": st.st_size,
                "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(),
                "kind": ("jsonl" if os.path.splitext(path)[1].lower() == ".jsonl" else "text"),
            }
        except Exception:
            meta = {"path": path, "size": None, "kind": "unknown"}

        def stream():
            yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"
            for chunk in _tail_sse(path, from_start=from_start, as_json=as_json):
                yield chunk

        return Response(stream(), mimetype="text/event-stream")

    return app


app = create_app()


if __name__ == "__main__":
    # Default host/port suitable for local use
    port = int(os.environ.get("PORT", "5000"))
    # Note: For Windows/VSCode, debug=True auto-reload may lock files; set as needed.
    app.run(host="127.0.0.1", port=port, debug=False)