# backend/Server.py
from flask import Flask, send_from_directory
from flask_cors import CORS

# Your helper that loads Credentials.json and returns a Firestore client
from config import initialize_firebase

app = Flask(__name__)
CORS(app)

# --- Firebase (using Credentials.json via config.py) ---
initialize_firebase()

# --- Register every blueprint in routes/__init__.py ---
from routes import register_blueprints
register_blueprints(app)

# --- Optional: print routes for sanity check ---
print("Registered routes:")
for rule in app.url_map.iter_rules():
    print(f"{rule.endpoint}: {rule}")

# --- Dev entry‑point (gunicorn ignores this block) ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
