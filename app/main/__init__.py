from flask import Blueprint

main_bp = Blueprint('main', __name__,
                    template_folder="../static/templates",
                    static_folder="../static",
                    static_url_path='/static')

from app.main import routes
