

from pyueye_example_camera import Camera
from pyueye_example_utils import FrameThread
from pyueye_example_gui import PyuEyeQtApp, PyuEyeQtView
from PyQt5 import QtCore, QtGui, QtWidgets
import configparser
import keyboard
import sys
from pyueye import ueye
import time
import cv2
import numpy as np
import matplotlib as mpl
mpl.use("QT4Agg")
import matplotlib.pyplot as plt
import sys

white_nMin1 = [(np.inf,np.inf,np.inf)]
first_image = True
previouse_step = [0,0,0]
RGB = [0,0,0]
std = [0,0,0]

DONE_WITH_EXPOSURE = False


def ustr(x):
    return x

def check_color_param(frame,x, y, r):
	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[y - r:y + r, x - r:x + r, :]

	# define range of blue color in HSV
	# lower_blue = np.array([60, 20,20])
	lower_blue = np.array([150, 20,20])
	# upper_blue = np.array([250,80,60])
	upper_blue = np.array([250,80,60])
	mask = cv2.inRange(hsv, lower_blue, upper_blue)

	# res = cv2.bitwise_and(hsv,hsv, mask=mask)
	# cv2.imshow(str(np.sum(mask==0)/(mask.shape[0]*mask.shape[1])),res)
	# cv2.imshow('frame',hsv)
	# cv2.imshow('s1', hsv)
	if np.sum(mask==0)/(mask.shape[0]*mask.shape[1]) <.90:
		cv2.imwrite(r"C:\Users\Oran\Desktop\whits\xyr_240X240_"+str(int(x))+str(int(y))+str(int(r))+'.tiff',  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[y - 170:y + 170, x - 170:x + 170, :])
		return True
	else:
		return False


def print_means(self, image_data, target=np.array([106,120,202])):#this values are for 100m channels new chip (plastic)
# def auto_adjust_color_and_exposure(self, image_data, target=np.array([141,157,196])):#this is for PDMS channel
	# print(type(self))
	image = image_data.as_1d_image()
	color_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	means = fast_mean(color_image)
	print(means)


	return QtGui.QImage(color_image.data,
                    image_data.mem_info.width,
                    image_data.mem_info.height,
                    QtGui.QImage.Format_RGB888)


def auto_adjust_color_and_exposure(self, image_data, target=np.array([106,120,202])):#this values are for 100m channels new chip (plastic)
# def auto_adjust_color_and_exposure(self, image_data, target=np.array([141,157,196])):#this is for PDMS channel
	# print(type(self))
	image = image_data.as_1d_image()
	color_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	means = fast_mean(color_image)
	print(str(means))

	delta = target-means
	print(str(delta))

	exposure_done = False

	pos = np.abs(delta) != np.min(np.abs(delta))
	if np.min(np.abs(delta)) < 4 and not np.any(delta[pos]<0):
		DONE_WITH_EXPOSURE = True
		print('deal_with_color')
		if np.any(np.abs(delta) > 5):
			update_config_gain(update = {['red','green','blue'][np.where(delta==np.max(delta))[0][0]]: '1'})
		else:
			print ('done')
			update_config_exposure(update=10)
			sys.exit()
	else:

		if np.any(delta < 0):
			exp = update_config_exposure(update=-1)


		elif np.min(np.abs(delta)) < 1:
			pass
		else:
			exp = update_config_exposure(update= 1)


	return QtGui.QImage(color_image.data,
	                    image_data.mem_info.width,
	                    image_data.mem_info.height,
	                    QtGui.QImage.Format_RGB888)



def fast_mean(image):
	hist0 = np.cumsum(cv2.calcHist([image], [0], None, [256], [0, 256]))
	hist1 = cv2.calcHist([image], [0], None, [256], [0, 256])
	hist2 = cv2.calcHist([image], [0], None, [256], [0, 256])

	# plt.figure('hist')
	# plt.plot(hist0)
	# plt.plot(hist1)
	# plt.plot(hist2)

	(r, g, b) = cv2.split(image)
	return np.array([r.mean(),g.mean(),b.mean()])
	# cv2.createCLAHE(image)todo check this as normalization for WBC
	#todo for concentration use cv2 hist method

def adjust_manually(self, image_data, x=None):
	image = image_data.as_1d_image()
	color_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


	r = np.mean(color_image[:,:,0])
	g = np.mean(color_image[:,:,1])
	b = np.mean(color_image[:,:,2])

	print( 'R:' + str(int(r))  +  '  G:' + str(int(b)) + '  B:' + str(int(g)))

	if keyboard.is_pressed('q'):
		update_config_gain(update = {'red':'1'})
	if keyboard.is_pressed('a'):
		update_config_gain( update ={'red':'-1'})

	if keyboard.is_pressed('w'):
		update_config_gain( update ={'green':'1'})
	if keyboard.is_pressed('s'):
		update_config_gain(update = {'green':'-1'})

	if keyboard.is_pressed('e'):
		update_config_gain(update ={'blue': '1'})
	if keyboard.is_pressed('d'):
		update_config_gain(update ={'blue': '-1'})


	return QtGui.QImage(color_image.data,
						image_data.mem_info.width,
						image_data.mem_info.height,
						QtGui.QImage.Format_RGB888)



def update_config_exposure(ini_file="/home/oran/Desktop/camera_params.ini",update = 0,set=False):
	config = configparser.ConfigParser()
	config.sections()
	config.read(ini_file, encoding='utf-8-sig')
	if set:
		config['Timing']['exposure'] = str(update)

	else:
		print('exposure set to ' +  str(int(config['Timing']['exposure']) + update))
		config['Timing']['exposure'] = str(int(config['Timing']['exposure']) + update)

	with open(ini_file, 'w') as configfile:
		config.write(configfile)

	pParam = ueye.wchar_p()
	pParam.value = ini_file
	ueye.is_ParameterSet(1, ueye.IS_PARAMETERSET_CMD_LOAD_FILE, pParam, 0)

	return config['Timing']['exposure']

def update_config_gain(ini_file="/home/oran/Pictures/Settings/default-camera-settings.ini",update = {'green':'10'},set=False):
	print('updating' + ini_file)
	config = configparser.ConfigParser()
	config.sections()
	config.read(ini_file, encoding='utf-8-sig')

	if set:
		for key in update.keys():
			config['Gain'][key] = update[key]
	else:
		for key in update.keys():
			config['Gain'][key] = str(int(config['Gain'][key]) + int(update[key]))
			# print(config['Gain'][key])
	with open(ini_file, 'w') as configfile:
		config.write(configfile)

	pParam = ueye.wchar_p()
	pParam.value = ini_file
	ueye.is_ParameterSet(1, ueye.IS_PARAMETERSET_CMD_LOAD_FILE, pParam, 0)

def main(config_path="/home/oran/Pictures/Settings/default-camera-settings.ini"):
	print(config_path)
	# we need a QApplication, that runs our QT Gui Framework
	app = PyuEyeQtApp()

	# a basic qt window
	view = PyuEyeQtView()
	view.resize(1920/1.5,1080/1.5)
	view.show()
	#update_config_gain(update={'red': '0','green' : '0','blue':'0'},set=True)
	#update_config_exposure(update=70,set=True)
	# view.user_callback = adjust_manually
	view.user_callback = print_means
	# view.user_callback = adjust_manually
	# camera class to simplify uEye API access
	cam = Camera()
	cam.init()
	# cam.set
	cam.set_colormode(ueye.IS_CM_BGR8_PACKED)
	pParam = ueye.wchar_p()
	pParam.value = config_path
	ueye.is_ParameterSet(1, ueye.IS_PARAMETERSET_CMD_LOAD_FILE, pParam, 0)

	# cam.set(cv2.cv.CV_CAP_PROP_EXPOSURE, 10)

	# cam.__getattribute__('is_CameraStatus')
	# cam.__setattr__('GetCameraInfo',0)
	#cam.set_aoi(0,0, 1280, 1024)
	cam.set_aoi(0,0, 4912 , 3684)
	cam.alloc()
	cam.capture_video()
	ueye._is_GetError
	ueye.is_Exposure(1, ueye.c_uint(1), ueye.c_void_p(),cbSizeOfParam=ueye.c_int(0))
	# ueye.IS_EXPOSURE_CMD_GET_FINE_INCREMENT_RANGE_MIN = 20
	# ueye.IS_EXPOSURE_CMD_GET_FINE_INCREMENT_RANGE_MAX = 21

	# a thread that waits for new images and processes all connected views
	thread = FrameThread(cam, view)
	thread.start()

	# update_config_gain()


	# cleanup
	app.exit_connect(thread.stop)
	app.exec_()

	thread.stop()
	thread.join()

	cam.stop_video()
	cam.exit()

if __name__ == "__main__":
	main()

