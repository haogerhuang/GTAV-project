import numpy as np
import cv2
import glob

def check_training_data(training_data,name):
	for i,j in training_data:

		cv2.imshow(name,i)
		print j
		cv2.waitKey(100)
list = glob.glob("*.npy")



for i,j in enumerate(list):
        
  training_data = np.load(j)
  check_training_data(training_data,str(i))		
