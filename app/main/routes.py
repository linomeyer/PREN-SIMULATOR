from flask import render_template
from app.main import main_bp


@main_bp.route('/')
def index():
    """Simple hello world endpoint."""
    return render_template("start.html")


@main_bp.route('/transformation')
def transformation():
    """Transformation page."""
    return render_template("transformation.html")


@main_bp.route('/erkennung')
def erkennung():
    """Erkennung page."""
    return render_template("erkennung.html")


@main_bp.route('/matching')
def matching():
    """Matching page."""
    return render_template("matching.html")
