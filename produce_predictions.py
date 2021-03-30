import tensorflow.compat.v1 as tf
import inputs as inputs
import model as inference
import scipy.io as sio
import config as cfg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def main(argv=None):
    while iterator.
    with tf.Graph().as_default():
        iterator, iterator_init = inputs.generate_iterator(cfg.FLAGS.filenames_path)
        next_image, outputPath = iterator.get_next()
        shape = tf.shape(next_image)

        prediction = inference.inference(next_image)
        print("___________")
        print(prediction)
        print(type(prediction))
        print("___________")
        prediction = cfg.FLAGS.bf / prediction
        restorePreviousState = tf.train.Saver(tf.global_variables())

        sess = tf.Session()
        sess.run([tf.global_variables_initializer(), iterator_init])
        restorePreviousState.restore(sess, cfg.FLAGS.chkpt_path)
        outputPred, outputName, _sh = sess.run([prediction, outputPath, shape])

        o = outputName[0]
        sio.savemat(o, {'data': outputPred})
        print("___________")
        print(outputPred)
        print("___________")
        print(outputPath)
        plt.imshow(outputPred[0], cmap='gray')
        im = Image.fromarray(np.squeeze((np.interp(outputPred[0], (outputPred[0].min(), outputPred[0].max()), (0, +255))).astype(np.uint8), axis=2))
        #im = Image.fromarray(outputPred.astype(np.uint8))
        im.save('/Users/sakshamramkhatod/Downloads/demo-final/outputs/output.png', format='png')
        plt.show()

if __name__ == '__main__':
    tf.app.run()
