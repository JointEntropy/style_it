import os
import tensorflow as tf
from .image_utils import load_image, preprocess_image, deprocess_image
from .squeezenet import SqueezeNet
from .components import gram_matrix, style_loss, content_loss, tv_loss
from tqdm import tqdm
from skimage.io import imsave



def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


tf.reset_default_graph()
sess = get_session()
SAVE_PATH = '/home/grigory/PycharmProjects/si/logic/squeezenet.ckpt'
if not os.path.exists(SAVE_PATH):
    raise ValueError("No squeezenet weights found!")
model = SqueezeNet(save_path=SAVE_PATH, sess=sess)

#Set up optimization hyperparameters
initial_lr = 3.0
decayed_lr = 0.1
decay_lr_at = 180
max_iter = 1000

PARAMS_COLLECTION = {
    'comp7': {  #
        'content_layer': 3,
        'content_weight': 0,#5e-2,
        'style_layers': (1, 4, 6, 7),
        'style_weights': (20000, 500, 12, 1),
        'tv_weight': 5e-2
    }
}


def style(content_image, style_image, image_size, style_size,
            content_layer, content_weight,
            style_layers, style_weights, tv_weight,
            update_each=10,
            save_each=10,
            lr=3.0,
            default_style_decay=0.1,
            default_content_weight=0.1,
            root_dir='',
            tmp_dir='tmp',
            init_random=False):
    """
    Inputs:
    - content_image: content image
    - style_image: style image
    - image_size: size of  image dimension (used for content loss and generated image)
    - style_size: size of style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    """
    content_weight_val = content_weight
    style_weights_val = style_weights
    content_weight = tf.Variable(content_weight, dtype=tf.float32, name='content_weights')

    style_weights = [tf.Variable(weight, dtype=tf.float32) for weight in style_weights]

    # Extract features from the content image
    content_img = preprocess_image(load_image(content_image, size=image_size))
    feats = model.extract_features(model.image)
    content_target = sess.run(feats[content_layer],
                              {model.image: content_img[None]})

    # Extract features from the style image
    style_img = preprocess_image(load_image(style_image, size=style_size))
    style_feat_vars = [feats[idx] for idx in style_layers]
    style_target_vars = []
    # Compute list of TensorFlow Gram matrices
    for style_feat_var in style_feat_vars:
        style_target_vars.append(gram_matrix(style_feat_var))
    # Compute list of NumPy Gram matrices by evaluating the TensorFlow graph on the style image
    style_targets = sess.run(style_target_vars, {model.image: style_img[None]} )

    # Initialize generated image to content image
    if init_random:
        img_var = tf.Variable(tf.random_uniform(content_img[None].shape, 0, 1), name="image")
    else:
        img_var = tf.Variable(content_img[None], name="image")

    # Extract features on generated image and compute content, style and variance losses
    feats = model.extract_features(img_var)
    c_loss = content_loss(content_weight, feats[content_layer], content_target)
    s_loss = style_loss(feats, style_layers, style_targets, style_weights)
    t_loss = tv_loss(img_var, tv_weight)
    loss = c_loss + s_loss + t_loss

    # Create and initialize the Adam optimizer
    lr_var = tf.Variable(initial_lr, name="lr")
    # Create train_op that updates the generated image when run
    with tf.variable_scope("optimizer") as opt_scope:
        train_op = tf.train.AdamOptimizer(lr_var).minimize(loss, var_list=[img_var])
    # Initialize the generated image and optimization variables
    opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)
    sess.run(tf.variables_initializer([lr_var, img_var, content_weight] + opt_vars))
    # Create an op that will clamp the image values when run
    clamp_image_op = tf.assign(img_var, tf.clip_by_value(img_var, -1.5, 1.5))

    style_weights_decay = 1
    for t in tqdm(range(max_iter)):
        # Take an optimization step to update img_var
        feed_dict = {
            content_weight: content_weight_val
        }
        for i, (weight_var, default_val) in enumerate(zip(style_weights, style_weights_val)):
            feed_dict[weight_var] = default_val*(style_weights_decay**(-i))
        sess.run(train_op, feed_dict)

        if t < decay_lr_at:
            sess.run(clamp_image_op)
        if t == decay_lr_at:
            sess.run(tf.assign(lr_var, decayed_lr))
        if t % update_each == 0:
            img = sess.run(img_var)
            img = deprocess_image(img[0], rescale=True)
            if t % save_each == 0:
                imsave(os.path.join(root_dir, tmp_dir, 'img_{}.jpg'.format(t)), img)
            new_params = yield img
            content_weight_val = new_params['content_weight']
            style_weights_decay = new_params['style_decay']


