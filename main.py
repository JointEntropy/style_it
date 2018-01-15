import os
import glob
from flask import Flask, request, render_template,   Response, abort, send_file, url_for
import base64
from logic.image_utils import imread
import imageio
import io
from logic.style_transfer import style, PARAMS_COLLECTION
import json
import numpy as np
from PIL import Image


app = Flask(__name__)
app.secret_key = 'Master Kenobi!'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

PARAMS = {
    'STYLE_MAX_SIDE': 500,
    'CONTENT_MAX_SIDE': 500
}


@app.route('/', methods=['GET'])
def welcome():
    return render_template('welcome.html')


processes = {}
images = {}


@app.route('/submit', methods=['POST'])
def submit():
    content_img = request.files.get('content')
    style_img = request.files.get('style')

    params = PARAMS_COLLECTION['comp7'].copy()
    params['content_weight'] = float(request.form.get('content_weight', 5e-2))

    output_gen = style(content_img, style_img,  PARAMS['CONTENT_MAX_SIDE'], PARAMS['STYLE_MAX_SIDE'], **params)
    # Возвращаем токен
    output_iter = iter(output_gen)
    token = abs(hash(output_iter))
    next(output_iter)
    processes[str(token)] = output_iter
    response = Response(json.dumps({'token': str(token)}), mimetype='text/json')
    return response


@app.route('/ping/', methods=['GET'])
def update_params():
    token = request.args['token']
    params = {
        'style_decay': float(request.args.get('style_decay', '0.5')),
        'content_weight': float(request.args.get('content_weight', '5e-2')),
    }
    img = processes[token].send(params)
    #imageio.imsave('output/kek.jpg', img, format='jpg')
    img_b = io.BytesIO()
    imageio.imsave(img_b, img, format='jpg')
    #response = send_file(img_b, mimetype='image/jpg')
    img_b.seek(0)
    image_64_encode = base64.encodebytes(img_b.read())
    response = Response(image_64_encode, mimetype='image/jpg')
    return response


@app.route('/stop/', methods=['GET'])
def stop_session():
    token = request.args.get('token')
    del processes[token]
    pass

if __name__ == '__main__':
    app.run()
    # response = Response(result, mimetype='text/json')
    # response.headers['Content-Disposition'] = "inline; filename=" + filename
