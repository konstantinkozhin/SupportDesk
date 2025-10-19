"""
run_with_ngrok.py

Cross-platform Python wrapper to start the app and expose it with ngrok.
- Starts the local server (uvicorn by default, or `start_server.bat`/`.sh` with --use-batch)
- Configures ngrok with the provided authtoken (if given)
- Starts an ngrok http tunnel to the chosen port and prints the public URL
- Cleans up child processes on Ctrl+C

Usage examples:
  python run_with_ngrok.py --token 32Pf3JDFeJTHcRoi0NkCqgUVV68_48P3gyynHQFuwqp8QJpC4
  python run_with_ngrok.py --use-batch

Requirements:
- Python 3.8+
- ngrok installed and on PATH (ngrok executable)
- Optional: uvicorn and project dependencies if you want the script to start uvicorn directly

Note: the script executes external programs; review it before running in sensitive environments.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
import json


def run():
    parser = argparse.ArgumentParser(description='Start local app and expose it via ngrok')
    parser.add_argument('--token', '-t', help='ngrok authtoken (optional, can be set via NGROK_AUTHTOKEN env var)')
    parser.add_argument('--port', '-p', type=int, default=8000, help='local port to expose (default: 8000)')
    parser.add_argument('--use-batch', action='store_true', help='use start_server.bat / start_server.sh instead of uvicorn')
    parser.add_argument('--wait', type=int, default=3, help='seconds to wait for server init before starting ngrok')

    args = parser.parse_args()
    token = args.token or os.environ.get('NGROK_AUTHTOKEN')
    port = args.port

    procs = []

    def terminate_all(sig=None, frame=None):
        print('\nStopping processes...')
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(0.5)
        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        sys.exit(0)

    signal.signal(signal.SIGINT, terminate_all)
    signal.signal(signal.SIGTERM, terminate_all)

    cwd = os.path.abspath(os.path.dirname(__file__))

    # Start local server
    if args.use_batch:
        # Try start_server.bat (Windows) or start_server.sh (Unix) depending on platform
        if sys.platform.startswith('win'):
            batch_path = os.path.join(cwd, 'start_server.bat')
            if not os.path.exists(batch_path):
                print('start_server.bat not found in repo root. Aborting.')
                return
            print('Starting server via start_server.bat...')
            p_server = subprocess.Popen([batch_path], cwd=cwd)
        else:
            sh_path = os.path.join(cwd, 'start_server.sh')
            if not os.path.exists(sh_path):
                print('start_server.sh not found in repo root. Aborting.')
                return
            print('Starting server via start_server.sh...')
            p_server = subprocess.Popen(['bash', sh_path], cwd=cwd)
    else:
        # Start uvicorn if available
        print(f'Starting uvicorn on 127.0.0.1:{port}...')
        # Prefer python -m uvicorn to respect venv
        cmd = [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', str(port)]
        try:
            p_server = subprocess.Popen(cmd, cwd=cwd)
        except FileNotFoundError:
            print('uvicorn not found. Install requirements or use --use-batch to start your server via script.')
            return

    procs.append(p_server)

    # Wait for the server to become reachable (basic wait)
    time.sleep(args.wait)

    # Setup ngrok
    ngrok_cmd = shutil_which('ngrok')
    if not ngrok_cmd:
        print('ngrok executable was not found in PATH. Please install ngrok and add to PATH.')
        terminate_all()

    if token:
        print('Configuring ngrok authtoken...')
        try:
            subprocess.run([ngrok_cmd, 'authtoken', token], check=True)
        except Exception as e:
            print('Failed to configure ngrok authtoken:', e)

    # Start ngrok
    print(f'Starting ngrok tunnel to localhost:{port}...')
    p_ngrok = subprocess.Popen([ngrok_cmd, 'http', str(port)], cwd=cwd)
    procs.append(p_ngrok)

    # Poll local ngrok API for tunnels
    api = 'http://127.0.0.1:4040/api/tunnels'
    public_url = None
    for i in range(40):
        try:
            with urllib.request.urlopen(api, timeout=1) as resp:
                data = json.load(resp)
                tunnels = data.get('tunnels', [])
                for t in tunnels:
                    if t.get('proto') == 'http':
                        public_url = t.get('public_url')
                        break
                if public_url:
                    break
        except Exception:
            time.sleep(0.5)
    if public_url:
        print('\nngrok public URL:', public_url)
        print('Tunnel will remain active while this script runs. Press Ctrl+C to stop.')
    else:
        print('Failed to obtain ngrok public URL. Check ngrok logs at http://127.0.0.1:4040')

    # Wait until terminated
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        terminate_all()


def shutil_which(cmd):
    """Simple cross-platform which() using shutil.which when available."""
    try:
        import shutil
        return shutil.which(cmd)
    except Exception:
        # fallback: naive search in PATH
        path = os.environ.get('PATH', '')
        exts = ['']
        if os.name == 'nt':
            exts = os.environ.get('PATHEXT', '').split(';')
        for d in path.split(os.pathsep):
            full = os.path.join(d, cmd)
            for e in exts:
                candidate = full + e
                if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                    return candidate
        return None


if __name__ == '__main__':
    run()
