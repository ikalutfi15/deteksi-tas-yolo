import csv
import os
import cv2
from flask import Flask, render_template, Response, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from ultralytics import YOLO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kunci-rahasia-skripsi-ta'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load model YOLOv8 best.pt
model = YOLO('best.pt')
cap = cv2.VideoCapture(0)


# Model Database User
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'admin' atau 'satpam'


# Model Database Kejadian
class Kejadian(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    waktu = db.Column(db.String(50), nullable=False)
    kamera = db.Column(db.String(100), nullable=False)
    id_tas = db.Column(db.String(30), nullable=False)
    jenis_tas = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    keputusan = db.Column(db.String(50), default="-")


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Inisialisasi Database & Akun Default (Admin & Satpam)
with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        admin_user = User(username='admin', password='123', role='admin')
        satpam_user = User(username='satpam', password='123', role='satpam')
        db.session.add(admin_user)
        db.session.add(satpam_user)
        db.session.commit()


def generate_frames():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Username atau Password salah!')
    return render_template('login.html')


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)


@app.route('/live-monitoring')
@login_required
def live_monitoring():
    return render_template('live_monitoring.html', user=current_user)


@app.route('/riwayat-kejadian')
@login_required
def riwayat_kejadian():
    data_db = Kejadian.query.order_by(Kejadian.id.desc()).all()

    csv_rows = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_csv = os.path.join(base_dir, "..", "kejadian", "log_kejadian.csv")
    if os.path.exists(log_csv):
        with open(log_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                csv_rows.append(row)

    return render_template(
        'riwayat_kejadian.html',
        user=current_user,
        kejadian_db=data_db,
        csv_rows=list(reversed(csv_rows)),
    )


@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)