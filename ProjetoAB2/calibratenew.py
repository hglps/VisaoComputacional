import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image

chessboard_size = (9,6)

obj_points = [] #3D points in real world space 
img_points = [] #3D points in image plane

# Grid of vertices

objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)

objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

# read images

calibration_paths = glob.glob('./camera_imgs/*')

for image_path in tqdm(calibration_paths):

	#Load image
	image = cv2.imread(image_path)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print("Checking corners in image...")
	#find chessboard corners
	ret,corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)

	if ret == True:
		print("Corners detected!")
		print(image_path)

		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		#refine corner location and accuracy
		cv2.cornerSubPix(gray_image, corners, (5,5), (-1,-1), criteria)
		obj_points.append(objp)
		img_points.append(corners)

# Camera calibration
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,gray_image.shape[::-1], None, None)

#Save parameters into numpy file
np.save("./camera_params/ret", ret)
np.save("./camera_params/K", K)
np.save("./camera_params/dist", dist)
np.save("./camera_params/rvecs", rvecs)
np.save("./camera_params/tvecs", tvecs)

#Get exif data in order to get focal length.
exif_img = PIL.Image.open(calibration_paths[0])

exif_data = {
	PIL.ExifTags.TAGS[k]:v
	for k, v in exif_img._getexif().items()
	if k in PIL.ExifTags.TAGS}

#Get focal length in tuple form
focal_length_exif = exif_data['FocalLength']
print('\n\nfocal length by exif: ',focal_length_exif,'\n')

#Get focal length in decimal form
focal_length = focal_length_exif

#Save focal length
np.save("./camera_params/FocalLength", focal_length)

print('Calibration finished!')
