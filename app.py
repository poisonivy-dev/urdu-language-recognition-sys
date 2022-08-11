import os
import cv2 as cv
import tensorflow as tf
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

app = Flask(__name__)


path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads\\')
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load Character set
with open('chars.txt', encoding='utf-8') as f:
    chars = f.read()


# Dense to corresponding text removing Unidentified Character
def dense_to_text(dense):
    text = ''
    for num in dense:
        if (num < len(chars) + 1 and num > 0):
            text += chars[num - 1]
    return text


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return "No image selected for uploading"
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + app.config['UPLOAD_FOLDER'] + filename)
        img_path = app.config['UPLOAD_FOLDER'] + filename
        image = cv.imread(img_path)
        # Loading Model
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session()
        model = tf.compat.v1.saved_model.loader.load(sess, tags=['serve'],
                                                     export_dir='model_pb')
        # Get Predicted Text
        resized_image = tf.compat.v1.image.resize_image_with_pad(image, 64, 1024).eval(session=sess)
        img_gray = cv.cvtColor(resized_image, cv.COLOR_RGB2GRAY).reshape(64, 1024, 1)

        output = sess.run('Dense-Decoded/SparseToDense:0',
                          feed_dict={
                              'Deep-CNN/Placeholder:0': img_gray
                          })
        output_text = dense_to_text(output[0])
        output_text = {'output_text':output_text}
        return jsonify(output_text)
    else:
        return "Allowed image types are - png, jpg, jpeg, gif"


if __name__ == '__main__':
    app.run(debug=True)
