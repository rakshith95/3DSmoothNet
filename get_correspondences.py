'''
Code to generate random keypoints, compute descriptors, and matches between two point clouds.
Usage:
python get_correspondences.py <Output_path>

correspondences.txt file is stored as <Output_path>correspondences.txt, so to store in same folder
python get_correpondences,py ./

'''

import tensorflow as tf
import copy
import numpy as np
import os
import subprocess
from open3d import *
import get_keypoints
from scipy.spatial import cKDTree
import sys


# Run the input parametrization
point_cloud_files = ["/home/rakshith/CTU/ARI/ari_ws/src/pointcloud_registration/pointClouds/PointCloudGood4.ply", "/home/rakshith/CTU/ARI/ari_ws/src/pointcloud_registration/pointClouds/cloud-rotated.ply"]
# point_cloud_files = ["/home/rakshith/CTU/ARI/ari_ws/src/pointcloud_registration/pointClouds/cloud-rotated.ply", "/home/rakshith/CTU/ARI/ari_ws/src/pointcloud_registration/pointClouds/PointCloudGood4.ply"]
output_keypoint_files = [point_cloud_files[0].split('.ply')[0]+'_keypoints.txt', point_cloud_files[1].split('.ply')[0]+'_keypoints.txt']
pcld_file1 = point_cloud_files[0].split('/')[-1]
pcld_file2 = point_cloud_files[1].split('/')[-1]


success = get_keypoints.get_keypoints(point_cloud_files[0], output_keypoint_files[0])
if not success:
    print('Keypoints file 1 already exists')
success = get_keypoints.get_keypoints(point_cloud_files[1], output_keypoint_files[1])
if not success:
    print('Keypoints file 2 already exists')

keypoints_files = [output_keypoint_files[0], output_keypoint_files[1]]

reference_file = None
test_file = None
import glob
try:
    npz_files = glob.glob('./data/demo/64_dim/*.npz')
    reference_file = [i for i in npz_files if pcld_file1 in i][0]
    test_file      = [i for i in npz_files if pcld_file2 in i][0]
    import os
    already_exists1 = os.path.exists(reference_file) 
    already_exists2 = os.path.exists(test_file) 
    if (already_exists1 and already_exists2):
        print('Files exist')
    else:
        raise Exception('Files don\'t exist') 
except Exception as E:
    print(E)
    for i in range(0,len(point_cloud_files)):
        args = "./build/./3DSmoothNet -f " + point_cloud_files[i] + " -k " + keypoints_files[i] +  " -o ./data/demo/sdv/"
        subprocess.call(args, shell=True)
    
    print('Input parametrization complete. Start inference')
    
    
    # Run the inference as shell 
    args = "python main_cnn.py --run_mode=test --evaluate_input_folder=./data/demo/sdv/  --evaluate_output_folder=./data/demo"
    subprocess.call(args, shell=True)
    print('Inference completed perform nearest neighbor search and registration')

    npz_files = glob.glob('./data/demo/64_dim/*.npz')
    reference_file = [i for i in npz_files if pcld_file1 in i][0]
    test_file      = [i for i in npz_files if pcld_file2 in i][0]


# Load the descriptors and estimate the transformation parameters using RANSAC
reference_desc = np.load(reference_file)
reference_desc = reference_desc['data']

test_desc = np.load(test_file)
test_desc = test_desc['data']

# Save as open3d feature 
ref = open3d.registration.Feature()
ref.data = reference_desc.T

test = open3d.registration.Feature()
test.data = test_desc.T

ref_kp_indices = np.loadtxt(keypoints_files[0])
test_kp_indices = np.loadtxt(keypoints_files[1])

ref = open3d.registration.Feature()
ref.data = reference_desc.T

test = open3d.registration.Feature()
test.data = test_desc.T

ref_feats_tree = cKDTree(ref.data.T)
distances, indices = ref_feats_tree.query(test.data.T, p=2, k=[1])
correspondences = np.hstack((ref_kp_indices[indices] ,test_kp_indices.reshape((-1,1))))

np.savetxt(sys.argv[1]+'correspondences.txt', correspondences)