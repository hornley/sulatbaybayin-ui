from flask import Flask, send_from_directory, request, jsonify, render_template
import os
import io
import base64
import tempfile
import traceback
import urllib.request
import threading
import time


try:
    from script.infer import load_checkpoint, draw_predictions
except Exception:
    # We'll handle missing module at runtime
    load_checkpoint = None
    draw_predictions = None

# Serve the existing frontend folder as static files.
app = Flask(
    __name__,
    template_folder='frontend/templates',
    static_folder='frontend/static')


# Try to import inference helpers

def find_checkpoint(root='.'):
    """Search for a .pth checkpoint file under the repo and return first match or None."""
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.pth'):
                return os.path.join(dirpath, fn)
    return None



# Try to locate and load a checkpoint at startup (if available). The path can be overridden
# by setting the INFER_CKPT environment variable.
MODEL = None
CLASSES = []
_MODEL_LOADED = False
# Determine checkpoint path: prefer explicit env var, otherwise try downloading from INFER_CKPT_URL,
# otherwise auto-discover a .pth under the repo.
CKPT_PATH = os.environ.get('INFER_CKPT')
if not CKPT_PATH:
    ckpt_url = os.environ.get('INFER_CKPT_URL')
    if ckpt_url:
        try:
            dest = os.path.join(tempfile.gettempdir(), 'infer_checkpoint.pth')
            print(f"Downloading checkpoint from {ckpt_url} -> {dest}")
            urllib.request.urlretrieve(ckpt_url, dest)
            CKPT_PATH = dest
        except Exception:
            print('Failed to download checkpoint from INFER_CKPT_URL:')
            traceback.print_exc()
    else:
        CKPT_PATH = find_checkpoint('.')

def _load_model_background(path):
    """Background model loader. Sets global MODEL, CLASSES and _MODEL_LOADED."""
    global MODEL, CLASSES, _MODEL_LOADED
    if not path:
        print('No checkpoint path provided to background loader.')
        return
    if load_checkpoint is None:
        print('load_checkpoint not available; cannot load model.')
        return
    try:
        print(f'Starting background model load from {path}')
        MODEL, CLASSES = load_checkpoint(path, device='cpu')
        _MODEL_LOADED = True
        print(f'Loaded checkpoint: {path}')
    except Exception:
        print('Failed to load checkpoint in background:')
        traceback.print_exc()


# Start background loading if a checkpoint is available
if CKPT_PATH:
    # kick off loader in a daemon thread so the process can accept requests immediately
    t = threading.Thread(target=_load_model_background, args=(CKPT_PATH,), daemon=True)
    t.start()
else:
    if CKPT_PATH is None:
        print('No checkpoint found automatically. Set INFER_CKPT or INFER_CKPT_URL to point to a .pth file.')
    if load_checkpoint is None:
        print('Inference helpers not available (missing script.infer).')


@app.route('/')
def index():
    """Renders the front page from templates/front.html."""
    # Ensure front.html is in your templates/ folder
    return render_template('front.html')

@app.route('/translate')
def translate():
    """Renders the translator page from templates/translate.html."""
    # Ensure translate.html is in your templates/ folder
    return render_template('translate.html')

@app.route('/alphabet')
def alphabet():
    """Renders the alphabet guide page from templates/alphabet.html."""
    # Ensure alphabet.html is in your templates/ folder
    return render_template('alphabet.html')



# API 
@app.route('/process_image', methods=['POST'])
def process_image():
    """Endpoint the frontend calls. Accepts multipart form 'image'. Runs detection and returns
    a JSON object with a data-URL of the annotated image and textual predictions.
    """
    if not _MODEL_LOADED or draw_predictions is None:
        return jsonify({'error': 'Model not loaded on server. Set INFER_CKPT to a valid .pth and restart.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request.'}), 400

    f = request.files['image']
    if f.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    # Save uploaded image to a temporary file because draw_predictions expects a path
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.filename)[1] or '.png')
    try:
        f.save(tmp.name)
        # run inference
        try:
            img_out, preds_sorted, best_sentence, overall_sentences, annotated_positions, raw_preds = draw_predictions(
                tmp.name, MODEL, CLASSES, device='cpu'
            )
        except Exception:
            traceback.print_exc()
            return jsonify({'error': 'Inference failed on server.'}), 500

        # build predictions text similar to script.infer main's per-image txt
        lines = []
        for x1, cname, score in preds_sorted:
            lines.append(f"{cname}\t{score:.4f}")
        lines.append('')
        lines.append('BEST_SENTENCE:')
        lines.append(best_sentence)
        lines.append('')
        lines.append('PERMUTATIONS:')
        for s in overall_sentences:
            lines.append(s)
        lines.append('')
        lines.append('POSITIONS (per-line, left->right, with candidates):')
        for li, line in enumerate(annotated_positions):
            lines.append(f'Line {li+1}:')
            for pi, pos in enumerate(line):
                labels = [pp['label'] for pp in sorted(pos, key=lambda r: r['score'], reverse=True)]
                lines.append(f'  Pos {pi+1}: ' + ','.join(labels))

        predictions_txt = "\n".join(lines)

        # convert PIL image to PNG bytes and then to data URL
        buffered = io.BytesIO()
        img_out.save(buffered, format='PNG')
        img_bytes = buffered.getvalue()
        b64 = base64.b64encode(img_bytes).decode('ascii')
        data_url = f'data:image/png;base64,{b64}'

        return jsonify({'output_url': data_url, 'translation': best_sentence, 'predictions_txt': predictions_txt})
    finally:
        try:
            tmp.close()
            os.unlink(tmp.name)
        except Exception:
            pass


@app.route('/health')
def health():
    """Simple health endpoint that reports whether the model is loaded."""
    return jsonify({
        'status': 'ok',
        'model_loaded': bool(_MODEL_LOADED),
        'ckpt_path': CKPT_PATH if 'CKPT_PATH' in globals() else None
    })


if __name__ == '__main__':
    # Use debug for local development. In production use a WSGI server.
    app.run(host='127.0.0.1', port=5000, debug=True)