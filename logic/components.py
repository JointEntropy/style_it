import tensorflow as tf


def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: scalar constant we multiply the content_loss by.
    - content_current: features of the current image, Tensor with shape [1, height, width, channels]
    - content_target: features of the content image, Tensor with shape [1, height, width, channels]

    Returns:
    - scalar content loss
    """
    # для content loss test
    _, H, W, C = content_current.shape
    # print('Default size:', H,W,C)
    # F = content_current.reshape([C,-1])
    F = tf.reshape(content_current, [C, -1])
    P = tf.reshape(content_original, [C, -1])
    # print('Unwrapped size', F.shape)
    return content_weight * tf.reduce_sum(tf.pow(F - P, 2))


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: Tensor of shape (1, H, W, C) giving features for
      a single image.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: Tensor of shape (C, C) giving the (optionally normalized)
      Gram matrices for the input image.
    """
    feat_shape = tf.shape(features)
    flatten = tf.reshape(features, [-1, feat_shape[-1]])
    features = tf.transpose(flatten)
    gram = tf.matmul(features, tf.transpose(features))
    if normalize:
        gram = gram / tf.reduce_prod(tf.cast(tf.shape(features), dtype=tf.float32))
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a Tensor giving the Gram matrix the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].

    Returns:
    - style_loss: A Tensor contataining the scalar style loss.
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be very much code (~5 lines). You will need to use your gram_matrix function.
    total_loss = None
    for l, style_source, w in zip(style_layers, style_targets, style_weights):
        style_current = gram_matrix(feats[l])
        layer_loss = w * tf.reduce_sum(tf.pow(style_current - style_source, 2))
        total_loss = total_loss + layer_loss if total_loss is not None else layer_loss
    return total_loss


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: Tensor of shape (1, H, W, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    img = tf.reshape(img, tf.shape(img)[1:])
    shift_size = tf.shape(img) - 1
    shift_size = tf.concat([shift_size[:-1], [3]], axis=0)
    width_shift = tf.slice(img, [1, 0, 0], shift_size)
    height_shift = tf.slice(img, [0, 1, 0], shift_size)
    trimmed = tf.slice(img, [0, 0, 0], shift_size)
    res = tf.pow(trimmed - width_shift, 2) + tf.pow(trimmed - height_shift, 2)
    return tv_weight * tf.reduce_sum(res)



