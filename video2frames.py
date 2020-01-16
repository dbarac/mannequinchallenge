import os
import cv2
import sys

if len(sys.argv) != 2:
	print('usage: python video2frames.py <path-to-video>')
	sys.exit()

vidpath = sys.argv[1]
vidcap = cv2.VideoCapture(vidpath)

f = open("test_data/test_user_video_list.txt", "w")

frames_path = "test_data/user_data/"
if not os.path.exists(frames_path):
	    os.makedirs(frames_path)
		
count = 0
success,image = vidcap.read()
while success:
	img_name = "{:05d}.jpg".format(count)
	cv2.imwrite(frames_path + img_name, image)     # save frame as JPEG file      
	f.write(frames_path + img_name + "\n")
	print('Read a new frame: {}'.format(count), success)
	count += 1
	success,image = vidcap.read()

f.close()

