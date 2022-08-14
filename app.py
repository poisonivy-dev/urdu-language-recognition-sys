import base64
import os
import cv2 as cv
import tensorflow as tf
from flask import Flask, jsonify, request, redirect, flash
from werkzeug.utils import secure_filename
from werkzeug.datastructures import ImmutableMultiDict
from scipy.ndimage import interpolation as inter
from PIL import Image as im
import numpy as np
from azure.storage.blob import BlobServiceClient

app = Flask(__name__)

# Setting Up Azure Blob

connect_str = "DefaultEndpointsProtocol=https;AccountName=ulrs2117345582;AccountKey=DVpEvHHNMCfrv/J3s6XK0NEvXqWAHurPkvFUdHdldgToHndiN9AnulueYWo1kjrNf+uFQrya747k+AStmjqtpA==;EndpointSuffix=core.windows.net"
container_name = "uploads"

# create a blob service client to interact with the storage account
blob_service_client = BlobServiceClient.from_connection_string(conn_str=connect_str)
try:
    # get container client to interact with the container in which images will be stored
    container_client = blob_service_client.get_container_client(container=container_name)
    # get properties of the container to force exception to be thrown if container does not exist
    container_client.get_container_properties()
except Exception as e:
    # create a container in the storage account if it does not exist
    container_client = blob_service_client.create_container(container_name)

# path = os.getcwd()
# UPLOAD_FOLDER = os.path.join(path, 'uploads\\')
# if not os.path.isdir(UPLOAD_FOLDER):
#     os.mkdir(UPLOAD_FOLDER)
#
# WORDS_UPLOAD_FOLDER = os.path.join(path, 'uploads\\words\\')
# if not os.path.isdir(WORDS_UPLOAD_FOLDER):
#     os.mkdir(WORDS_UPLOAD_FOLDER)
#
# LINES_UPLOAD_FOLDER = os.path.join(path, 'uploads\\lines\\')
# if not os.path.isdir(LINES_UPLOAD_FOLDER):
#     os.mkdir(LINES_UPLOAD_FOLDER)

app.secret_key = "secret key"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['LINES_UPLOAD_FOLDER'] = LINES_UPLOAD_FOLDER
# app.config['WORDS_UPLOAD_FOLDER'] = WORDS_UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


# Convert result from blob to numpy array of bytes
def blob_to_array(blob):
    arr = np.asarray(bytearray(blob), dtype=np.uint8)
    return arr


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


# Converting image to a binary image
def binary_otsus(image, filter: int = 1):
    # """Binarize an image 0's and 255's using Otsu's Binarization"""

    if len(image.shape) == 3:
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # Otsus Binarization
    if filter != 0:
        blur = cv.GaussianBlur(gray_img, (3, 3), 0)
        binary_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    else:
        binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    # Morphological Opening
    # kernel = np.ones((3,3),np.uint8)
    # clean_img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)

    return binary_img


def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score


def deskew(binary_img):
    ht, wd = binary_img.shape
    # _, binary_img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    # pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = (binary_img // 255.0)

    delta = 0.1
    limit = 3
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    # print('Best angle: {}'.formate(best_angle))

    # correct skew
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    img = im.fromarray((255 * data).astype("uint8"))

    # img.save('skew_corrected.png')
    pix = np.array(img)
    return pix


def save_image(img, title):
    filename = secure_filename(f'{title}.png')
    # cv.imwrite(f'{folder}/{title}.png', img)
    # Upload the resized image
    _, img_encode = cv.imencode('.png', img)
    img_upload = img_encode.tobytes()
    container_client.upload_blob(filename, img_upload, overwrite=True)


def read_image(title):
    filename = f'{title}.png'
    blob = container_client.download_blob(filename).readall()
    x = blob_to_array(blob)
    # decode the array into an image
    image = cv.imdecode(x, cv.IMREAD_UNCHANGED)
    return image


def projection(gray_img, axis: str = 'horizontal'):
    """ Compute the horizontal or the vertical projection of a gray image """

    if axis == 'horizontal':
        projection_bins = np.sum(gray_img, 1).astype('int32')
    elif axis == 'vertical':
        projection_bins = np.sum(gray_img, 0).astype('int32')

    return projection_bins


def preprocess(image):
    # Maybe we end up using only gray level image.
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_img = cv.bitwise_not(gray_img)

    binary_img = binary_otsus(gray_img, 0)
    # cv.imwrite('origin.png', gray_img)

    # deskewed_img = deskew(binary_img)
    deskewed_img = deskew(binary_img)
    # cv.imwrite('output.png', deskewed_img)

    # binary_img = binary_otsus(deskewed_img, 0)
    # breakpoint()

    # Visualize

    # breakpoint()
    return deskewed_img


def projection_segmentation(clean_img, axis, cut=3):
    segments = []
    start = -1
    cnt = 0

    projection_bins = projection(clean_img, axis)
    for idx, projection_bin in enumerate(projection_bins):

        if projection_bin != 0:
            cnt = 0
        if projection_bin != 0 and start == -1:
            start = idx
        if projection_bin == 0 and start != -1:
            cnt += 1
            if cnt >= cut:
                if axis == 'horizontal':
                    segments.append(clean_img[max(start - 1, 0):idx, :])
                elif axis == 'vertical':
                    segments.append(clean_img[:, max(start - 1, 0):idx])
                cnt = 0
                start = -1

    return segments


def line_horizontal_projection(image, cut=3):
    # Preprocess input image
    clean_img = preprocess(image)

    # Segmentation
    lines = projection_segmentation(clean_img, axis='horizontal', cut=cut)

    return lines


def word_vertical_projection(line_image, cut=3):
    line_words = projection_segmentation(line_image, axis='vertical', cut=cut)
    line_words.reverse()


def extract_words(img, visual=0):
    lines = line_horizontal_projection(img)
    words = []

    for idx, line in enumerate(lines):

        if visual:
            save_image(line, f'line{idx}')

    #     line_words = word_vertical_projection(line)
    #     for w in line_words:
    #         # if len(words) == 585:
    #         #     print(idx)
    #         words.append((w, line))
    #     # words.extend(line_words)
    #
    # # breakpoint()
    # if visual:
    #     for idx, word in enumerate(words):
    #         save_image(word[0], WORDS_UPLOAD_FOLDER, f'word{idx}')

    return lines


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('no file found')
        return redirect(request.url)
    # data = dict(request.form)
    # img = data['content']
    file = request.files['file']
    # filename = secure_filename('uploadedImage.png')
    if file.filename == "":
        flash('no image selected')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # file.save(os.path.join(app.config['WORDS_UPLOAD_FOLDER'], filename))
        # file.save(os.path.join(app.config['LINES_UPLOAD_FOLDER'], filename))
        # Uploading file in azure
        # upload the file to the container using the filename as the blob name
        container_client.upload_blob(filename, file, overwrite=True)
        print('uploaded')

        # Fetching the file url
        blob = container_client.download_blob(filename).readall()
        x = blob_to_array(blob)
        # decode the array into an image
        image = cv.imdecode(x, cv.IMREAD_UNCHANGED)

        # print(img_path)
        # image = cv.imread(img_path)

        # segmentation
        lines = extract_words(image, 1)
        print('lines upload complete')
        # Loading Model
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session()
        model = tf.compat.v1.saved_model.loader.load(sess, tags=['serve'],
                                                     export_dir='model_pb')
        output_text = ""
        for idx, line in enumerate(lines):
            image = read_image(f'line{idx}')
            image = image.reshape(image.shape[0], image.shape[1], 1)
            print('fetched resized image')
            # Get Predicted Text
            resized_image = tf.compat.v1.image.resize_image_with_pad(image, 64, 1024).eval(session=sess)
            # img_gray = cv.cvtColor(resized_image, cv.COLOR_RGB2GRAY).reshape(64, 1024, 1)
            print('added padding')
            output = sess.run('Dense-Decoded/SparseToDense:0',
                              feed_dict={
                                  'Deep-CNN/Placeholder:0': resized_image
                              })
            print('got output')
            out = dense_to_text(output[0])
            output_text += out
            print(output_text)
        output_text = {'output_text': output_text}
        return jsonify(output_text)
    else:
        return "Allowed image types are - png, jpg, jpeg, gif"


if __name__ == '__main__':
    app.run(debug=True)
