"""
pyngrok_runner.py

Simple runner that starts the local FastAPI/uvicorn app and opens a pyngrok tunnel.
Usage:
  # inside your activated venv
  pip install pyngrok uvicorn
  python pyngrok_runner.py --port 8000 --token YOUR_NGROK_TOKEN

The script:
 - starts `python -m uvicorn app.main:app --host 127.0.0.1 --port <port>` in a subprocess
 - uses pyngrok to open a public tunnel and prints the URL
 - on Ctrl+C it closes the tunnel and stops the server subprocess

Note: If you already run the server separately, pass --no-server to only open a tunnel.
"""

import argparse
import subprocess
import sys
import time
import signal

try:
    from pyngrok import ngrok
except Exception as e:
    print("pyngrok is required. Install with: pip install pyngrok")
    raise


def main():
    parser = argparse.ArgumentParser(description='Run uvicorn locally and expose via pyngrok')
    parser.add_argument('--port', '-p', type=int, default=8000, help='local port (default 8000)')
    parser.add_argument('--token', '-t', help='ngrok authtoken (optional)')
    parser.add_argument('--no-server', action='store_true', help='do not start uvicorn; assume it runs already')
    args = parser.parse_args()

    port = args.port
    server_proc = None

    def stop_all(signum=None, frame=None):
        print('\nStopping...')
        try:
            ngrok.kill()
        except Exception:
            pass
        if server_proc and server_proc.poll() is None:
            try:
                server_proc.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    if not args.no_server:
        cmd = [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', str(port)]
        print('Starting uvicorn:', ' '.join(cmd))
        server_proc = subprocess.Popen(cmd, cwd='.')
        # give server some time to boot
        time.sleep(2)

    if args.token:
        try:
            ngrok.set_auth_token(args.token)
        except Exception as e:
            print('Failed to set ngrok token:', e)

    print('Opening pyngrok tunnel to http://127.0.0.1:%d ...' % port)
    try:
        tunnel = ngrok.connect(port)
    except Exception as e:
        print('pyngrok failed to open tunnel:', e)
        stop_all()

    public_url = tunnel.public_url
    print('\nPublic URL:', public_url)
    print('Press Ctrl+C to stop and close the tunnel')

    # Wait until process ends or interrupted
    try:
        while True:
            time.sleep(1)
            if server_proc and server_proc.poll() is not None:
                print('Server process exited with code', server_proc.returncode)
                break
    except KeyboardInterrupt:
        pass

    stop_all()


if __name__ == '__main__':
    main()
