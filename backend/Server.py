import os
import json
from flask import Flask, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials

# Load local .env only if not in production
if os.getenv("ENV") != "production":
    load_dotenv()  # Loads .env in local dev

# --- Firebase Initialization ---
def initialize_firebase():
    if firebase_admin._apps:
        return  # Already initialized

    firebase_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
    if not firebase_json:
        raise RuntimeError(
            "FIREBASE_CREDENTIALS_JSON env var missing. "
            "Set it in .env (local) or Render Dashboard (prod)."
        )

    cred_dict = json.loads(firebase_json)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

# --- App Factory ---
def create_app():
    static_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "frontend", "build"
    ))
    app = Flask(__name__, static_folder=static_dir, static_url_path="/")
    CORS(app)

    # Init Firebase
    initialize_firebase()

    # Register blueprints
    from routes import register_blueprints
    register_blueprints(app)

    # Optional: Serve React build
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_react(path):
        if path and os.path.exists(os.path.join(static_dir, path)):
            return send_from_directory(static_dir, path)
        return send_from_directory(static_dir, "index.html")

    return app

# Gunicorn entrypoint
app = create_app()

# Dev mode runner
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
