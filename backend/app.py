from pathlib import Path
from threading import Thread
from flask import Flask, request, make_response
from werkzeug.utils import secure_filename
from flask import Flask, request
from flask_cors import CORS
from flask import Flask, jsonify, request
from pyrogram import Client
import os
import sys
sys.path.append(os.path.abspath("../aihandler"))
from liver_trauma_detection import ai_main_func  # nopep8

WORKINGDIR = os.path.abspath("../aiworkingdir")
UPLOAD_FOLDER = os.path.join(WORKINGDIR, "uploads")
ALLOWED_EXTENSIONS = {'zip'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, supports_credentials=True)

api_id = 1614783
api_hash = '65b3a547d9d2f0b6145a8ad3896e6313'

client = Client(session_name='myclient', api_id=api_id, api_hash=api_hash)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload-dicom', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'dicomzip' not in request.files:
            return jsonify(status="false")
        file = request.files['dicomzip']
        if file.filename == '':
            return jsonify(status="false")
        if file and allowed_file(file.filename):
            uploadFolderPath = Path(app.config['UPLOAD_FOLDER'])
            if not uploadFolderPath.exists():
                os.mkdir(uploadFolderPath)
            uploadedFile = os.path.join(
                app.config['UPLOAD_FOLDER'], "dicom_study.zip")
            file.save(uploadedFile)
            thread = Thread(target=ai_main_func,
                            args=(WORKINGDIR, uploadedFile))
            thread.start()
            return jsonify(status="true")


@app.route('/send-telegram/<username>', methods=['GET', 'POST'])
def send_telegram(username):
    print(username)
    os.system(f'python3 send.py "{username}"')
    return jsonify(msg="ok"), 200


@app.route('/download-report', methods=['GET', 'POST'])
def download():
    downloadFile = Path(WORKINGDIR) / "report_final" / "report.zip"
    if downloadFile.exists():
        f = open(downloadFile, "rb")
        f.seek(0)
        response = make_response(f.read())
        response.headers.set('Content-Type', 'zip')
        response.headers.set('Content-Disposition', 'attachment', filename='%s' %
                             os.path.basename(downloadFile))
        os.remove(downloadFile)
        return response
    else:
        return "Report is not ready yet"


def create_app():
    return app
