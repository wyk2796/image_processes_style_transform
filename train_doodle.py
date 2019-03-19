import tensorflow as tf
from model.doodle import DoodleModel
import os, time
import params as P


with tf.Session() as session:
    log_dir = P.doodle_logs + "/Monet{}".format(time.time())
    model = DoodleModel(session)
    model.create_network()
    model.loss_function()
    writer = tf.summary.FileWriter(log_dir, session.graph)
    model.write_summary = writer
    saver = tf.train.Saver()
    model.train(50)
    saver.save(session, P.doodle_saved_model_dir)

