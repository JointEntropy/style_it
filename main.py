import sys
from flask import Flask, request, render_template,   Response
import base64
import imageio
import io
from logic.style_transfer import style, PARAMS_COLLECTION
import json
import argparse
from pathlib import Path


app = Flask(__name__)
app.secret_key = 'Master Kenobi!'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def init():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--default-style-decay', default=0.5, type=float)
    arg('--default-content-weight', default=5e-2, type=float)
    arg('--content-max-side', type=int, default='450', help='Maximum width of content image.')
    arg('--style-max-side', type=int, default='450', help='Maximum width of style image.')
    arg('--lr', type=float, default=3.0)
    arg('--tmp-dir', type=str, default='tmp', help='Directory for saving each iteration results.')
    arg('--save-each', type=int, default='10', help='Save results step.')
    arg('--update-each', type=int, default='10', help='Update result  step.')
    arg('--root-dir', type=str, default='')
    # arg('--model', type=str, default='UNet', choices=['UN', 'UNet11'])

    args = parser.parse_args()
    # Создаём директорию для сохранения результатов на каждой итерации стилизации
    root = Path(args.tmp_dir)
    root.mkdir(exist_ok=True, parents=True)
    return vars(args)


@app.route('/', methods=['GET'])
def welcome():
    return render_template('welcome.html')


@app.route('/submit', methods=['POST'])
def submit():
    content_img = request.files.get('content')
    style_img = request.files.get('style')

    params = PARAMS_COLLECTION['comp7'].copy()
    params['content_weight'] = float(request.form.get('content_weight', 5e-2))

    output_gen = style(content_img, style_img,
                       PARAMS['content_max_side'],
                       PARAMS['style_max_side'],
                       update_each=PARAMS['update_each'],
                       save_each=PARAMS['save_each'],
                       tmp_dir=PARAMS['tmp_dir'],
                       lr=PARAMS['lr'],
                       **params)
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
    img_b = io.BytesIO()
    imageio.imsave(img_b, img, format='jpg')
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
    processes = {}
    images = {}
    PARAMS = init()
    app.run()
