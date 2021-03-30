import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS
tf.app.flags.DEFINE_integer(
    'input_image_channels', 3, 
    """Number of channels in the input image.""")
tf.app.flags.DEFINE_integer(
    'inference_image_height', 188, 
    """Height of the image for inference.""")
tf.app.flags.DEFINE_integer(
    'inference_image_width', 621, 
    """Width of the image for inference.""")

tf.app.flags.DEFINE_string(
    'filenames_path', 
    "/Users/sakshamramkhatod/Downloads/demo-final/filenames.txt", 
    """Path to file contatining input-output names.""")
tf.app.flags.DEFINE_string('chkpt_path', 
"/Users/sakshamramkhatod/Downloads/model/depth-chkpt", 
"""Path to saved model without extension""")

tf.app.flags.DEFINE_float('bf', 
359.7176277195809831 * 0.54, 
"""Baseline times focal length""")