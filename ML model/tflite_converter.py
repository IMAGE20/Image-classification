import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt

# freeze.py
model = keras.models.load_model('practice13_sgd_momentum.h5', compile=False)

export_path = './model_pb2'
model.save(export_path, save_format='tf')


# convert.py
saved_model_dir = './model_pb2'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('./model_pb2/converted_model2.tflite', 'wb').write(tflite_model)
