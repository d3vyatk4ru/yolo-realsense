
import numpy as np
import cv2
from realsense_camera import *

CLASSES = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def load_yolo():

	net = cv2.dnn.readNet("yolo\\yolov3-tiny.weights", "yolo\\yolov3-tiny.cfg")
	output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
	colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	return net, CLASSES, colors, output_layers

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width, depth_frame):
	boxes = []
	confs = []
	class_ids = []
	distance_mm = 0

	for output in outputs:
		for detect in output:

			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]

			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
				distance_mm = depth_frame[center_y, center_x]
				
	return boxes, confs, class_ids, distance_mm

model, classes, colors, output_layers = load_yolo()

# Load Realsense camera
rs = RealsenseCamera()

while True:
	# Get frame in real time from Realsense camera
	ret, bgr_frame, depth_frame = rs.get_frame_stream()
	height, width, _ = bgr_frame.shape
	blob, outputs = detect_objects(bgr_frame, model, output_layers)

	boxes, confs, class_ids, distance_mm = get_box_dimensions(outputs, height, width, depth_frame)

	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			# x, y - уоординаты левого верхнего угла
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), color, 2)
			cv2.putText(bgr_frame, label + f'-- {distance_mm} mm', (x, y - 5), font, 1, color, 1)

	# Видео, чтобы постмотреть
	cv2.imshow("Video stream", bgr_frame)

	key = cv2.waitKey(1)
	if key == 27:
		break

rs.release()
cv2.destroyAllWindows()
