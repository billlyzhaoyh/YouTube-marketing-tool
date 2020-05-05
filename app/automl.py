import numpy as np
import tensorflow as tf

# Load TFLite model
def load_vision(model_path='./models/automl/model.tflite'):

	interpreter = tf.lite.Interpreter(model_path=model_path)
	interpreter.allocate_tensors()

	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	return interpreter,input_details,output_details