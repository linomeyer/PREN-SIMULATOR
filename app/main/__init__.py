from flask import Blueprint

main_bp = Blueprint('main', __name__, template_folder="static/templates", static_folder="static")

from app.main import routes
