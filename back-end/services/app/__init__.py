from app import app, db
from models import User
from auth import *


db.init_app(app)
with app.app_context():
    # db.drop_all()
    db.create_all()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
