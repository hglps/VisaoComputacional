import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
import json
import dist as calcDist

def getDataPoints(filepath):

	filename = 'left_keypoints.json'

	with open(filename, 'rb') as file:
		data_pose = json.load(file)

	data_pose_points = data_pose['people'][0]['pose_keypoints_2d']
	data_points = []
	for i in range(0,75,3):
		data_points.append([data_pose_points[i], data_pose_points[i+1]])
	
	return data_points

def generate_output(vertices, colors, filename):
		colors = colors.reshape(-1,3)
		vertices = np.hstack([vertices.reshape(-1,3),colors])
	
		ply_header = '''ply
			format ascii 1.0
			element vertex %(vert_num)d
			property float x
			property float y
			property float z
			property uchar red
			property uchar green
			property uchar blue
			end_header
			'''
		with open(filename, 'w') as file:
			file.write(ply_header %dict(vert_num=len(vertices)))
			np.savetxt(file,vertices,'%f %f %f %d %d %d')

def gen_disparity_map(img1_path, img2_path, maxDisp=6, block_size=5, uniqueness_ratio=5, speckle_window_size=50, speckle_range=1):
	"""
	return disparity_map, output_points, output_colors
	"""

	ret = np.load('./camera_params/ret.npy', allow_pickle=True)
	K = np.load('./camera_params/K.npy',allow_pickle=True)
	dist = np.load('./camera_params/dist.npy',allow_pickle=True)

	img_1 = cv2.imread(img1_path)
	img_2 = cv2.imread(img2_path)

	# Height and weight
	h,w = img_2.shape[:2]

	new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))
	img_1_undistorted = cv2.undistort(img_1, K, dist, None, new_camera_matrix)
	img_2_undistorted = cv2.undistort(img_2, K, dist, None, new_camera_matrix)

	#Disparity object
	win_size = 3
	min_disp = 0
	max_disp = 16*int(maxDisp) #min_disp * 9
	num_disp = max_disp - min_disp # Needs to be divisible by 16

	stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
		numDisparities = num_disp,
		blockSize = int(block_size),
		uniquenessRatio = int(uniqueness_ratio),
		speckleWindowSize = int(speckle_window_size),
		speckleRange = int(speckle_range),
		disp12MaxDiff = 2,
		P1 = 8*3*win_size**2,
		P2 =32*3*win_size**2)

	# Compute disparity map
	print ("\nGenerating the disparity map")
	disparity_map = stereo.compute(img_1_undistorted, img_2_undistorted)

	focal_length = np.load('./camera_params/FocalLength.npy',allow_pickle=True)

	# Perspective transformation matrix
	Q2 = np.float32([[1,0,0,0],
					[0,-1,0,0],
					[0,0,focal_length*0.1,0], #Focal length multiplication obtained experimentally. 
					[0,0,0,1]])

	points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)

	# Color points
	colors = cv2.cvtColor(img_1_undistorted, cv2.COLOR_BGR2RGB)
	mask_map = disparity_map > disparity_map.min()

	output_points = points_3D[mask_map]
	output_colors = colors[mask_map]

	return disparity_map, output_points, output_colors

def reconstruct_3d(flag, output_points, output_colors, max_threshold=200):
	# Points from skeleton (OpenPose)

	data_points = getDataPoints('')

	body_output_points = []
	body_output_colors = []

	# Segments from OpenPose:
	# 0-1-2-3-4
	# 1-5-6-7
	# 0-15-17
	# 0-16-18

	# Index of output points and colors
	j = 0

	if flag == 'full':
		print('\nGenerating 3d cloud output file')

		output_file = 'recons_left_full.ply'
		generate_output(output_points, output_colors, output_file)
		print('\n\nDone!')
		status = 'File ' + output_file + ' created!'

		return status, body_output_points, body_output_colors

	# Selecting points from body
	print('Selecting points from body')

	for point in output_points:
		if output_colors[j].tolist() == [0,0,0]:
			j +=1
			continue

		segment = ''
		threshold = int(max_threshold) # max distance accepted

		# 0-1-2-3-4
		for i in range(4):
			x1 = data_points[i][0]
			y1 = data_points[i][1]
			x2 = data_points[i+1][0]
			y2 = data_points[i+1][1]

			shortest_dist = 1000000
			shortest_point = []
			dist_to_body = calcDist.dist(x1, y1, x2, y2, point[0], point[1]*(-1))
			if dist_to_body < shortest_dist:
				shortest_dist = dist_to_body
				shortest_point = point
				segment = str(i) + ', ' + str(i+1)

		# 1-5	
		x1 = data_points[1][0]
		y1 = data_points[1][1]
		x2 = data_points[5][0]
		y2 = data_points[5][1]

		dist_to_body = calcDist.dist(x1, y1, x2, y2, point[0], point[1]*(-1))
		if dist_to_body < shortest_dist:
			shortest_dist = dist_to_body
			shortest_point = point
			segment = '1, 5'

		# 0-15
		x1 = data_points[0][0]
		y1 = data_points[0][1]
		x2 = data_points[15][0]
		y2 = data_points[15][1]

		dist_to_body = calcDist.dist(x1, y1, x2, y2, point[0], point[1]*(-1))
		if dist_to_body < shortest_dist:
			shortest_dist = dist_to_body
			shortest_point = point
			segment = '0, 15'

		# 0-16
		x1 = data_points[0][0]
		y1 = data_points[0][1]
		x2 = data_points[16][0]
		y2 = data_points[16][1]
		dist_to_body = calcDist.dist(x1, y1, x2, y2, point[0], point[1]*(-1))
		if dist_to_body < shortest_dist:
			shortest_dist = dist_to_body
			shortest_point = point
			segment = '0, 16'

		# 5-6-7 
		for i in range(5,7,1):
			x1 = data_points[i][0]
			y1 = data_points[i][1]
			x2 = data_points[i+1][0]
			y2 = data_points[i+1][1]

			dist_to_body = calcDist.dist(x1, y1, x2, y2, point[0], point[1]*(-1))
			if dist_to_body < shortest_dist:
				shortest_dist = dist_to_body
				shortest_point = point
				segment = str(i) + ', ' + str(i+1)

		# 15-17 / 16-18
		for i in range(15,16,1):
			x1 = data_points[i][0]
			y1 = data_points[i][1]
			x2 = data_points[i+2][0]
			y2 = data_points[i+2][1]

			dist_to_body = calcDist.dist(x1, y1, x2, y2, point[0], point[1]*(-1))
			if dist_to_body < shortest_dist:
				shortest_dist = dist_to_body
				shortest_point = point
				segment = str(i) + ', ' + str(i+2)


		# Threshold
		if shortest_dist <= threshold:
			body_output_points.append(point)
			body_output_colors.append(output_colors[j])

		j +=1

	body_output_points = np.array(body_output_points)
	body_output_colors = np.array(body_output_colors)

	print('\nGenerating 3d cloud output file')

	output_file = 'recons_left_' + str(threshold) + '.ply'
	generate_output(body_output_points, body_output_colors, output_file)
	print('\n\nDone!')
	status = 'File ' + output_file + ' created!'

	return status, body_output_points, body_output_colors
