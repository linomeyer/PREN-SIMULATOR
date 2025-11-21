from flask import Blueprint

puzzle_gen_bp = Blueprint('puzzle_gen', __name__,
                          url_prefix='/puzzle-gen')

from app.puzzle_gen import routes
