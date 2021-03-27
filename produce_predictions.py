import tensorflow.compat.v1 as tf
import inputs as inputs
import model as inference
import scipy.io as sio
import config as cfg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def main(argv=None):
    with tf.Graph().as_default():
        iterator, iterator_init = inputs.generate_iterator(cfg.FLAGS.filenames_path)
        next_image, outputPath = iterator.get_next()
        shape = tf.shape(next_image)

        prediction = inference.inference(next_image)
        prediction = cfg.FLAGS.bf / prediction
        restorePreviousState = tf.train.restorePreviousState(tf.global_variables())

        sess = tf.Session(config=config)
        sess.run([tf.global_variables_initializer(), iterator_init])
        restorePreviousState.restore(sess, cfg.FLAGS.chkpt_path)
        outputPred, outputName, _sh = sess.run([prediction, outputPath, shape])

        o = outputName[0]
        sio.savemat(o, {'data': outputPred})
        print("___________")
        print(outputPred)
        print("___________")
        plt.imshow(outputPred[0], cmap='gray')
        plt.show()

if __name__ == '__main__':
    tf.app.run()
