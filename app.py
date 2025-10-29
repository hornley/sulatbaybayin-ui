from flask import Flask, send_from_directory, request, jsonify
import os
import io
import base64
import tempfile
import traceback

# Try to import inference helpers
try:
    from script.infer import load_checkpoint, draw_predictions
except Exception:
    # We'll handle missing module at runtime
    load_checkpoint = None
    draw_predictions = None

# Serve the existing frontend folder as static files.
app = Flask(__name__, static_folder='frontend', static_url_path='')


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
CKPT_PATH = "C:\\Users\\buend\\Documents\\GitHub\\sulat-baybayin-ui\\models\\best_seen.pth"
if CKPT_PATH and load_checkpoint is not None:
    try:
        # load to CPU by default (script handles cuda selection internally but we force cpu map)
        MODEL, CLASSES = load_checkpoint(CKPT_PATH, device='cpu')
        _MODEL_LOADED = True
        print(f"Loaded checkpoint: {CKPT_PATH}")
    except Exception:
        print('Failed to load checkpoint:')
        traceback.print_exc()
else:
    if CKPT_PATH is None:
        print('No checkpoint found automatically. Set INFER_CKPT env var to point to a .pth file.')
    if load_checkpoint is None:
        print('Inference helpers not available (missing script.infer).')


@app.route('/')
def index():
    # Serve the frontend/index.html
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def static_proxy(path):
    """Serve a file from the frontend folder if it exists, otherwise fall back to index.html.

    This supports single-page apps that use client-side routing.
    """
    frontend_dir = os.path.join(app.root_path, app.static_folder)
    file_path = os.path.join(frontend_dir, path)
    if os.path.isfile(file_path):
        return send_from_directory(frontend_dir, path)
    # fallback to index.html
    return app.send_static_file('index.html')


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


if __name__ == '__main__':
    # Use debug for local development. In production use a WSGI server.
    app.run(host='127.0.0.1', port=5000, debug=True)
