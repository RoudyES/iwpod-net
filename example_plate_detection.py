import numpy as np
from src.keras_utils import load_model
import cv2
from src.keras_utils import  detect_lp_width
from src.utils 					import  im2single
from src.drawing_utils			import draw_losangle
import argparse
import tensorflow as tf
import os
import shutil

def predict(imgPath, args, outputName = '1', outputPath = 'output'):
		Ivehicle = cv2.imread(imgPath)
		vtype = args.vtype
		iwh = np.array(Ivehicle.shape[1::-1],dtype=float).reshape((2,1))

		if (vtype in ['car', 'bus', 'truck']):
			#
			#  Defines crops for car, bus, truck based on input aspect ratio (see paper)
			#
			ASPECTRATIO = max(1, min(2.75, 1.0*Ivehicle.shape[1]/Ivehicle.shape[0]))  # width over height
			WPODResolution = 256# faster execution
			lp_output_resolution = tuple(ocr_input_size[::-1])

		elif  vtype == 'fullimage':
			#
			#  Defines crop if vehicles were not cropped 
			#
			ASPECTRATIO = 1 
			WPODResolution = 480 # larger if full image is used directly
			lp_output_resolution =  tuple(ocr_input_size[::-1])
		else:
			#
			#  Defines crop for motorbike  
			#
			ASPECTRATIO = 1.0 # width over height
			WPODResolution = 208
			lp_output_resolution = (int(1.5*ocr_input_size[0]), ocr_input_size[0]) # for bikes, the LP aspect ratio is lower

		#
		#  Runs IWPOD-NET. Returns list of LP data and cropped LP images
		#
		Llp, LlpImgs,_ = detect_lp_width(iwpod_net, im2single(Ivehicle), WPODResolution*ASPECTRATIO, 2**4, lp_output_resolution, lp_threshold)
		for i, img in enumerate(LlpImgs):
			#
			#  Draws LP quadrilateral in input image
			#
			pts = Llp[i].pts * iwh
			draw_losangle(Ivehicle, pts, color = (0,0,255.), thickness = 2)
			#
			#  Shows each detected LP
			#
			#cv2.imshow('Rectified plate %d'%i, img )
		#
		#  Shows original image with deteced plates (quadrilateral)
		#
		if not os.path.exists(outputPath):
			os.makedirs(outputPath, exist_ok=False)
		cv2.imwrite(outputPath + '/' + outputName + '.jpg', Ivehicle)
		#cv2.waitKey()
		#cv2.destroyAllWindows()

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i' 		,'--image'			,type=str   , default = 'images\\example_aolp_fullimage.jpg'		,help='Input Image')
	parser.add_argument('-v' 		,'--vtype'			,type=str   , default = 'fullimage'		,help = 'Image type (car, truck, bus, bike or fullimage)')
	parser.add_argument('-t' 		,'--lp_threshold'	,type=float   , default = 0.35		,help = 'Detection Threshold')
	parser.add_argument('-m' 		,'--model'	,type=str   , default = 'weights/my_trained_iwpod'		,help = 'Model Path')

	#parser.add_argument('-tr'		,'--train-dir'		,type=str   , required=True		,help='Input data directory for training')
	args = parser.parse_args()

	#
	#  Parameters of the method
	#
	#lp_threshold = 0.35 # detection threshold
	lp_threshold = args.lp_threshold
	ocr_input_size = [80, 240] # desired LP size (width x height)
	
	#
	#  Loads network and weights
	#
	iwpod_net = load_model(args.model)
	#iwpod_net = tf.keras.models.load_model(args.model)
	

	#
	#  Loads image with vehicle crop or full image with vehicle(s) roughly framed.
	#  You can use your favorite object detector here (a fine-tuned version of Yolo-v3 was
	#  used in the paper)
	#
	
	#
	#  Also inform the vehicle type:
	#  'car', 'bus', 'truck' 
	#  'bike' 
	#  'fullimage' 
	#
	#
	if os.path.isdir(args.image):
		for i, filename in enumerate(os.listdir(args.image)):
			f = os.path.join(args.image, filename)
			predict(f, args, str(i))
	else:
		predict(args.image, args)
		