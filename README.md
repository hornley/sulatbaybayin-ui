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

## Deploying on Railway (free tier friendly)

This project can be deployed to Railway using the included `Dockerfile` and `Procfile`.

High-level steps:

1. Create a Railway account and new project (connect your GitHub repository or push your code to Railway via the CLI).
2. In Railway, choose "Deploy from GitHub" and pick this repository.
3. In Railway project settings -> Variables, add one of these environment variables:
	- `INFER_CKPT` — a path inside the container to a `.pth` file (if you add the model to the repo; not recommended for large files), OR
	- `INFER_CKPT_URL` — a public URL (HTTP/HTTPS) pointing to your `.pth` checkpoint. The app will download this file at startup.

Example: set `INFER_CKPT_URL` to `https://my-bucket.s3.amazonaws.com/models/best.pth`.

4. Railway will build the Docker image using the `Dockerfile` in this repo. The Dockerfile installs system deps, Python deps from `requirements.txt`, and CPU PyTorch.

Notes and tips:
- PyTorch wheels are large — expect longer build times on the first deploy.
- The Dockerfile installs CPU-only PyTorch. If you require GPU, you'll need a provider and plan with GPU support (Railway does not provide GPUs on free tier).
- If your model is large, host it externally (S3, Google Cloud Storage) and set `INFER_CKPT_URL` in Railway — the app will download it at startup.
- Monitor memory usage: choose a Railway plan with enough RAM to load your model.

After deployment, Railway provides a public URL; your frontend will be served there and the `/process_image` endpoint will accept image uploads.

If you'd like, I can also:
- Add a small environment-check endpoint that returns model load status.
- Provide a sample `docker-compose.yml` for local testing with `INFER_CKPT_URL`.

