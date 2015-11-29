import tensorflow as tf
import numpy as np
from models import VGG16, I2V
from utils import read_image, parseArgs, getModel, add_mean, sub_mean
import argparse

content_image_path, params_path, modeltype, maxfilters = parseArgs()

print "Read images..."
content_image_raw = read_image(content_image_path)
content_image = sub_mean(content_image_raw)

with tf.Graph().as_default(), tf.Session() as sess:
    print "Load content values..."
    image = tf.constant(content_image)
    model = getModel(image, params_path, modeltype)
    content_image_y_val = [sess.run(y_l) for y_l in model.y()]  # sess.run(y_l) is a constant numpy array
    
    # Set up the summary writer (saving summaries is optional)
    # (do `tensorboard --logdir=/tmp/vgg-visualizer-logs` to view it)
    with tf.variable_scope("Input"):
        tf.image_summary("Input Image", content_image)
    for l, y in enumerate(content_image_y_val):
        print "Layer ", l, " : ", y.shape
        with tf.variable_scope("Layer_%d"%l):
            for i in range(y.shape[3]):
                if i >= maxfilters:
                    break
                temp = np.zeros((1, y.shape[1], y.shape[2], 1)).astype(np.float32)
                temp[0,:,:,0] = y[0,:,:,i]
                tf.image_summary("Layer %d, Filter %d"%(l, i), tf.constant(temp, name="Layer_%d_Filter_%d"%(l, i)))    
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('/tmp/vgg-visualizer-logs', graph_def=sess.graph_def)
    
    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, 0)
