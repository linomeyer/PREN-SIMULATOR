from flask import Flask


def create_app():
    """Flask application factory."""
    app = Flask(__name__)

    # Register blueprints
    from app.main import main_bp
    app.register_blueprint(main_bp)

    return app
