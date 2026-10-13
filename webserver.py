#!/usr/bin/env python3
"""Live plot gallery, using only the Python standard library.

Run ``python webserver.py`` beside your lecture scripts, then open port 8000
in a browser. Any ``plt.savefig('plots.png')`` in this directory appears
automatically. Overwriting the file updates its existing card.
Each card's delete button removes the image file from this directory.

An optional directory and port can be supplied:
    python webserver.py ./plots --port 8000

Only images directly inside the chosen directory are served. Subdirectories,
hidden files, symlinks and source files are excluded.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


IMAGE_TYPES = {
    '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
    '.svg': 'image/svg+xml', '.webp': 'image/webp', '.gif': 'image/gif',
}

PAGE = r'''<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live plots</title>
<style>
:root { color-scheme: light; --ink:#36444B; --muted:#65757D; --teal:#2B8983;
  --line:#DDE5E9; --paper:#F7F9FA; --red:#C0392B; }
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink);
  font:16px/1.55 "Open Sans","DejaVu Sans",sans-serif; }
main { max-width:1440px; margin:auto; padding:40px 32px; }
header { display:flex; align-items:center; justify-content:space-between;
  flex-wrap:wrap; gap:24px; margin-bottom:30px; }
h1 { margin:0; font-size:30px; font-weight:600; letter-spacing:-.6px; }
header p { margin:4px 0 0; color:var(--muted); }
.controls { display:flex; align-items:center; gap:20px; }
#status { font-size:14px; color:var(--teal); }
#status::before { content:""; display:inline-block; width:7px; height:7px;
  margin-right:8px; border-radius:50%; background:currentColor; }
#status[data-state="paused"] { color:var(--muted); }
#status[data-state="offline"] { color:var(--red); }
button { font:inherit; font-size:14px; color:var(--ink); cursor:pointer;
  background:white; border:1px solid var(--line); border-radius:6px; padding:8px 14px; }
button:hover { border-color:var(--teal); color:var(--teal); }
button:focus-visible { outline:3px solid var(--teal); outline-offset:4px; }
.summary { display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap;
  padding:14px 0; border-top:1px solid var(--line); color:var(--muted); font-size:13px; }
#grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr)); gap:24px; }
.card { min-width:0; overflow:hidden; background:white; border:1px solid var(--line); border-radius:10px; }
.card-head { padding:16px 20px 0; display:flex; align-items:center; gap:12px; }
.card-title { flex:1; min-width:0; }
h2 { font-size:17px; font-weight:600; margin:0; overflow-wrap:anywhere; }
time { font-size:12px; color:var(--muted); white-space:nowrap; }
.delete { display:flex; align-items:center; justify-content:center; flex-shrink:0;
  width:36px; height:36px; padding:8px; color:var(--muted); }
.delete:hover { color:var(--red); border-color:var(--red); }
.delete:disabled { opacity:.45; cursor:wait; }
.delete-error { margin:8px 20px 0; color:var(--red); font-size:13px; }
.figure { display:flex; align-items:center; justify-content:center; width:100%;
  min-height:160px; padding:12px; }
.figure img { display:block; max-width:100%; max-height:540px; height:auto; object-fit:contain; }
.loading { color:var(--muted); padding:70px 0; }
#empty { text-align:center; padding:100px 24px; background:white;
  border:1px dashed var(--line); border-radius:10px; }
#empty h2 { font-size:22px; }
#empty p { color:var(--muted); }
code { font-family:"DejaVu Sans Mono",monospace; font-size:13px; }
#empty code { display:inline-block; padding:12px 18px; background:var(--paper); border-radius:6px; }
[hidden] { display:none !important; }
@media (max-width:600px) {
  main { padding:24px 16px; } header { gap:16px; } h1 { font-size:26px; }
}
</style>
<main>
  <header>
    <div><h1>Live plots</h1><p>Figures, as you save them.</p></div>
    <div class="controls">
      <span id="status" role="status">Connecting</span>
      <button id="pause" type="button" aria-pressed="false">Pause updates</button>
    </div>
  </header>
  <div class="summary"><span id="count">0 figures</span></div>
  <div id="empty" hidden>
    <h2>Ready for your first plot</h2>
    <p>Save a figure in the watched folder. It will appear here automatically.</p>
    <code>plt.savefig('plots.png')</code>
  </div>
  <div id="grid"></div>
</main>
<script>
const grid = document.querySelector('#grid');
const empty = document.querySelector('#empty');
const status = document.querySelector('#status');
const pause = document.querySelector('#pause');
const cards = new Map();
let paused = false, refreshing = false, revision = 0;

function setStatus(label, state = '') {
  status.textContent = label;
  status.dataset.state = state;
}

function updateSummary() {
  document.querySelector('#count').textContent = `${cards.size} figure${cards.size === 1 ? '' : 's'}`;
  empty.hidden = cards.size !== 0;
}

async function deletePlot(name, card) {
  card.remove.disabled = true;
  card.error.hidden = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(`/plots/${encodeURIComponent(name)}`, {
      method:'DELETE', signal:controller.signal,
    });
    if (!response.ok && response.status !== 404) throw new Error(response.status);
    revision++;  // Ignore any poll that started before this deletion completed.
    card.node.remove();
    if (cards.get(name) === card) cards.delete(name);
    updateSummary();
  } catch (error) {
    card.error.textContent = 'Could not delete the file. Try again.';
    card.error.hidden = false;
  } finally {
    clearTimeout(timeout);
    card.remove.disabled = false;
  }
}

function createCard(name) {
  const node = document.createElement('section');
  node.className = 'card';
  node.innerHTML = '<div class="card-head"><div class="card-title"><h2></h2><time></time></div>' +
    '<button class="delete" type="button"><svg width="18" height="18" viewBox="0 0 24 24" ' +
    'fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>' +
    '</svg></button></div><p class="delete-error" role="alert" hidden></p>' +
    '<div class="figure"><span class="loading">Saving figure…</span></div>';
  node.querySelector('h2').textContent = name;
  const remove = node.querySelector('.delete');
  remove.setAttribute('aria-label', `Delete ${name}`);
  remove.title = `Delete ${name} from disk`;
  const card = {node, remove, figure:node.querySelector('.figure'), time:node.querySelector('time'),
    error:node.querySelector('.delete-error'), version:null, pending:false};
  remove.addEventListener('click', () => deletePlot(name, card));
  return card;
}

function updateImage(card, plot) {
  if (card.version === plot.version || card.pending) return;
  card.pending = true;
  const image = new Image();
  image.alt = plot.name;
  // Load the replacement first: a half-written save must not blank the old plot.
  image.onload = () => {
    card.pending = false;
    if (paused || cards.get(plot.name) !== card) return;
    card.figure.replaceChildren(image);
    card.version = plot.version;
    const date = new Date(plot.modified * 1000);
    card.time.dateTime = date.toISOString();
    card.time.textContent = date.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  };
  image.onerror = () => { card.pending = false; };
  image.src = plot.url;
}

async function refresh() {
  if (paused || refreshing) return;
  refreshing = true;
  const startedAtRevision = revision;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch('/api/plots', {cache:'no-store', signal:controller.signal});
    if (!response.ok) throw new Error(response.status);
    const plots = await response.json();
    if (paused || startedAtRevision !== revision) return;
    const names = new Set(plots.map(plot => plot.name));
    for (const [name, card] of cards) {
      if (!names.has(name)) {
        card.node.remove();
        cards.delete(name);
      }
    }
    plots.forEach((plot, index) => {
      if (!cards.has(plot.name)) cards.set(plot.name, createCard(plot.name));
      const card = cards.get(plot.name);
      if (grid.children[index] !== card.node) grid.insertBefore(card.node, grid.children[index] || null);
      updateImage(card, plot);
    });
    updateSummary();
    setStatus('Live');
  } catch (error) {
    if (!paused) setStatus('Reconnecting…', 'offline');
  } finally {
    clearTimeout(timeout);
    refreshing = false;
  }
}

pause.addEventListener('click', () => {
  paused = !paused;
  pause.textContent = paused ? 'Resume updates' : 'Pause updates';
  pause.setAttribute('aria-pressed', String(paused));
  setStatus(paused ? 'Paused' : 'Connecting', paused ? 'paused' : '');
  if (!paused) refresh();
});
refresh();
setInterval(refresh, 1000);
</script>
</html>
'''.encode('utf-8')


def image_path(root, name):
    """Accept a visible, regular image directly inside the watched directory."""
    if not name or name.startswith('.') or '/' in name or '\\' in name:
        raise FileNotFoundError(name)
    path = root / name
    if path.suffix.lower() not in IMAGE_TYPES or path.is_symlink() or not path.is_file():
        raise FileNotFoundError(name)
    return path


def list_plots(root):
    plots = []
    for candidate in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        try:
            path = image_path(root, candidate.name)
            stat = path.stat()
        except OSError:
            continue  # Files may be replaced or deleted while a figure is saving.
        version = f'{stat.st_mtime_ns:x}-{stat.st_ctime_ns:x}-{stat.st_size:x}'
        plots.append({
            'name': path.name,
            'version': version,
            'modified': stat.st_mtime,
            'url': f'/plots/{quote(path.name, safe="")}?v={version}',
        })
    return plots


def make_handler(root):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond()

        def do_HEAD(self):
            self.respond(head=True)

        def do_DELETE(self):
            route = unquote(urlsplit(self.path).path)
            if not route.startswith('/plots/'):
                self.send_error(404)
                return
            try:
                image_path(root, route[len('/plots/'):]).unlink()
            except (FileNotFoundError, ValueError):
                self.send_error(404)
                return
            except OSError:
                self.send_error(500, 'Could not delete the image file')
                return
            self.send_response(204)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()

        def respond(self, head=False):
            route = unquote(urlsplit(self.path).path)
            try:
                if route == '/':
                    content, content_type = PAGE, 'text/html; charset=utf-8'
                elif route == '/api/plots':
                    content = json.dumps(list_plots(root)).encode('utf-8')
                    content_type = 'application/json; charset=utf-8'
                elif route.startswith('/plots/'):
                    path = image_path(root, route[len('/plots/'):])
                    content, content_type = path.read_bytes(), IMAGE_TYPES[path.suffix.lower()]
                else:
                    self.send_error(404)
                    return
            except (OSError, ValueError):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if content_type == 'image/svg+xml':
                self.send_header('Content-Security-Policy', "sandbox; default-src 'none'; style-src 'unsafe-inline'")
            self.end_headers()
            if not head:
                try:
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, format, *args):
            pass  # Keep the lecture terminal clear of polling requests.

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', nargs='?', default='.', type=Path, help='folder containing the figures (default: current folder)')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='0.0.0.0', help='bind address (default: all interfaces)')
    args = parser.parse_args()
    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'Not a directory: {root}')
    if not 0 <= args.port <= 65535:
        parser.error('Port must be between 0 and 65535')
    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(root))
    except OSError as error:
        parser.exit(1, f'Cannot start server: {error}\n')
    host = 'localhost' if args.host == '0.0.0.0' else args.host
    print(f'Live plots: http://{host}:{server.server_port}/', flush=True)
    print(f'Watching {root} (Ctrl-C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
