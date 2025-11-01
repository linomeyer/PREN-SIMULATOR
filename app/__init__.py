
from flask import Flask
import os


def create_app():
    """Flask application factory."""
    app = Flask(__name__,
                template_folder='static/templates',
                static_folder='static')

    # Create necessary directories
    os.makedirs('app/static/output', exist_ok=True)
    os.makedirs('app/main/img', exist_ok=True)

    # Register blueprints
    from app.main import main_bp
    app.register_blueprint(main_bp)

    return app