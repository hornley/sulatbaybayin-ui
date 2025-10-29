# sulatbaybayin
Baybayin Translator

## Backend (Flask)

This repository contains a minimal Flask backend that serves the static `frontend` folder.

Files added:
- `app.py` — small Flask app that serves `frontend/index.html` at `/` and serves other static files.
- `requirements.txt` — lists Flask dependency.

### Run locally (Windows, cmd.exe)

1. Create and activate a virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the app:

```
python app.py
```

Open http://127.0.0.1:5000/ in your browser.

Alternatively, you can set FLASK_APP and use `flask run`:

```
set FLASK_APP=app.py
flask run
```

### Inference model (optional)

The Flask app can run the detection/translation pipeline from `script/infer.py` if a PyTorch checkpoint is available.

- To point the server at a checkpoint, set the `INFER_CKPT` environment variable to the `.pth` file path before starting the app. Example (cmd.exe):

```
set INFER_CKPT=C:\path\to\your\model\best.pth
python app.py
```

- The server will try to auto-discover a `.pth` file under the repository if `INFER_CKPT` is not set.

- Note: inference requires PyTorch / torchvision and appropriate GPU drivers if you want GPU acceleration. The `requirements.txt` currently lists only Flask; install PyTorch separately (see https://pytorch.org/) or add it to `requirements.txt` if you want it managed by pip.

