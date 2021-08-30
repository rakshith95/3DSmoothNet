import sys
import numpy as np 
import open3d as o3d
import os

def get_keypoints(input_pcld_file, output_keypoints_file):
    
    already_exists = os.path.exists(output_keypoints_file) 
    if not already_exists:
        input_filename = input_pcld_file
        # output_filename = input_filename.split('.ply')[0]+'_keypoints.txt'
        output_filename = output_keypoints_file
        print("Saving file: ", output_filename)    
        pcld = o3d.io.read_point_cloud(input_filename)
        num_pts = np.asarray(pcld.points).shape[0] 
        with open(output_filename, "w") as f:
            for i in range(10000):
                index = np.random.randint(0, num_pts)
                f.writelines(str(index))
                f.writelines('\n')
        return True
    else:
        return False
