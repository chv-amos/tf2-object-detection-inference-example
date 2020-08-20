import numpy as np
import tensorflow as tf
import cv2
import pathlib
import time

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# TO DO: Change the relative path to the models repository folder
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# TO DO: Change model name based on the usrl in: 
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
model_name = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'

# Frame counting variables
frame_count = 0
frame_speed = 0
time_now = 0
time_prev = 0

def load_model(model_name):
	base_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
	model_file = model_name + '.tar.gz'
	model_dir = tf.keras.utils.get_file(
		fname=model_name, 
		origin=base_url + model_file,
		untar=True)

	model_dir = pathlib.Path(model_dir)/"saved_model"
	model = tf.saved_model.load(str(model_dir))

	return model

def run_inference_for_single_image(model, image):
	image = np.asarray(image)
	# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
	input_tensor = tf.convert_to_tensor(image)
	# The model expects a batch of images, so add an axis with `tf.newaxis`.
	input_tensor = input_tensor[tf.newaxis,...]

	# Run inference
	model_fn = model.signatures['serving_default']
	output_dict = model_fn(input_tensor)

	# All outputs are batches tensors.
	# Convert to numpy arrays, and take index [0] to remove the batch dimension.
	# We're only interested in the first num_detections.
	num_detections = int(output_dict.pop('num_detections'))
	output_dict = {key:value[0, :num_detections].numpy() 
								 for key,value in output_dict.items()}
	output_dict['num_detections'] = num_detections

	# detection_classes should be ints.
	output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
	 
	# Handle models with masks:
	if 'detection_masks' in output_dict:
		# Reframe the the bbox mask to the image size.
		detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
							output_dict['detection_masks'], output_dict['detection_boxes'],
							 image.shape[0], image.shape[1])      
		detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,tf.uint8)
		output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
		
	return output_dict

def run_inference(model, cap):
	global frame_speed, frame_count, time_now, time_prev
	while cap.isOpened():
		ret, image_np = cap.read()

		# Actual detection.
		output_dict = run_inference_for_single_image(model, image_np)
		
		# Visualization of the results of a detection.
		vis_util.visualize_boxes_and_labels_on_image_array(
			image_np,
			output_dict['detection_boxes'],
			output_dict['detection_classes'],
			output_dict['detection_scores'],
			category_index,
			instance_masks=output_dict.get('detection_masks_reframed', None),
			use_normalized_coordinates=True,
			line_thickness=8)

		cv2.putText(image_np, 'FPS: {:.2f}'.format(frame_speed), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,50),2)
		cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
		
		# Calculate FPS
		frame_count += 1
		time_now = time.time()	
		if time_now - time_prev >= 1:
			frame_speed = frame_count/(time_now - time_prev)
			time_prev = time_now
			frame_count = 0

		if cv2.waitKey(1) & 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			break


detection_model = load_model(model_name)

cap = cv2.VideoCapture(0) # or cap = cv2.VideoCapture("<video-path>")

run_inference(detection_model, cap)

