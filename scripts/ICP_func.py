#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import open3d as o3d
import copy
import csv
#from matplotlib import pyplot as plt
from open3d import visualization


# In[ ]:


#to save point clouds for ICP from csv file
def save_ply(pts, write_text, filename):
    pcd = o3d.geometry.PointCloud()

    # the method DoubleVector() will convert float64 numpy array of shape (n,) to Open3D format.
    # see http://www.http://www.open3d.org/docs/release/python_api/open3d.utility.DoubleVector.html
    pcd.points = o3d.utility.Vector3dVector(pts)

    # http://www.open3d.org/docs/release/python_api/open3d.io.write_point_cloud.html#open3d.io.write_point_cloud
    o3d.io.write_point_cloud(filename, pcd, write_ascii=write_text)
    
    # read ply file
    pcd = o3d.io.read_point_cloud(filename)    
    #print('file saved to:',filename)


# In[ ]:


import time
def draw_registration_result(source, target, transformation, filename):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.update_geometry(source_temp)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename)
    vis.destroy_window()

