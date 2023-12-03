import open3d as o3d

print(o3d.__version__)

import os
import copy
import numpy as np
import pandas as pd
from PIL import Image

np.random.seed(42)

def get_pcd_mmc_coords(pcd, coords_type="centre"):
    
    """Get the centre, minimum or maximum XYZ coordinates of a 3D point cloud.
    
    Args:
        pcd: An open3d.geometry.PointCloud object.
        coords_type: (string, default="centre") Either "centre", "min" or "max" to get the corresponding XYZ coordinates.
        
    Returns:
        (array) An array of the corresponding XYZ coordinates from the 3D point cloud.
    """
    
    if coords_type == "centre":
        return pcd.get_center()
    elif coords_type == "max":
        return pcd.get_max_bound()
    elif coords_type == "min":
        return pcd.get_min_bound()


def get_hpr_camera_radius(pcd):
    
    """Obtain the camera and radius parameters to define the camera viewpoint for the hidden point removal operation
    based on the dimensions of the 3D point cloud.
    
    Args:
        pcd: An open3d.geometry.PointCloud object.
        
    Returns:
        camera: (list of floats) A list of the corresponding XYZ coordinates for the camera position.
        radius: (float) The radius parameter for the camera viewpoint.
    """
    
    diameter = np.linalg.norm(np.asarray(get_pcd_mmc_coords(pcd, "min")) - np.asarray(get_pcd_mmc_coords(pcd, "max")))
    camera = [0, 0, diameter]
    radius = diameter * 100
    
    return camera, radius



def get_rotated_pcd(pcd, x_theta, y_theta, z_theta):
    """Defining a function to rotate a point cloud in X, Y and Z-axis."""
    pcd_rotated = copy.deepcopy(pcd)
    R = pcd_rotated.get_rotation_matrix_from_axis_angle([x_theta, y_theta, z_theta])
    pcd_rotated.rotate(R, center=(0, 0, 0))
    
    return pcd_rotated

def get_hpr_pt_map(pcd, camera, radius):

    _, pt_map = pcd.hidden_point_removal(camera, radius)    
    return pt_map


if __name__=='__main__':

    # Defining the path to the 3D model file.

    mesh_path = "data/3d_model.obj"


    # Reading the 3D model file as a 3D mesh using open3d.

    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # Computing the normals for the mesh.

    mesh.compute_vertex_normals()
    draw_geoms_list = [mesh]
    o3d.visualization.draw_geometries(draw_geoms_list)


    # Creating a mesh of the XYZ axes Cartesian coordinates frame.
    mesh_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])

    draw_geoms_list = [mesh_coord_frame, mesh]
    o3d.visualization.draw_geometries(draw_geoms_list)

    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_points = np.asarray(bbox.get_box_points())
    bbox_points[:, 2] = np.clip(bbox_points[:, 2], a_min=None, a_max=0)
    bbox_cropped = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_points))
    mesh_cropped = mesh.crop(bbox_cropped)

    draw_geoms_list = [mesh_coord_frame, mesh_cropped]
    o3d.visualization.draw_geometries(draw_geoms_list)


    n_pts = 100_000
    pcd = mesh.sample_points_uniformly(n_pts)

    draw_geoms_list = [mesh_coord_frame, pcd]
    o3d.visualization.draw_geometries(draw_geoms_list)

    pcd_cropped = pcd.crop(bbox_cropped)

    draw_geoms_list = [mesh_coord_frame, pcd_cropped]
    o3d.visualization.draw_geometries(draw_geoms_list)


    # Defining the camera and radius parameters for the hidden point removal operation.

    diameter = np.linalg.norm(np.asarray(pcd.get_min_bound()) - np.asarray(pcd.get_max_bound()))
    camera = [0, 0, diameter]
    radius = diameter * 100

    print(camera)
    print(radius)

    _, pt_map = pcd.hidden_point_removal(camera, radius)

    # Painting all the visible points in the point cloud in blue, and all the hidden points in red.

    pcd_visible = pcd.select_by_index(pt_map)
    pcd_visible.paint_uniform_color([0, 0, 1])    # Blue points are visible points (to be kept).
    print("No. of visible points : ", pcd_visible)

    pcd_hidden = pcd.select_by_index(pt_map, invert=True)
    pcd_hidden.paint_uniform_color([1, 0, 0])    # Red points are hidden points (to be removed).
    print("No. of hidden points : ", pcd_hidden)


    def deg2rad(deg):
        return deg * np.pi/180

    # Rotating the point cloud about the X-axis by 90 degrees.

    x_theta = deg2rad(90)
    y_theta = deg2rad(0)
    z_theta = deg2rad(0)

    tmp_pcd_r = copy.deepcopy(pcd)
    R = tmp_pcd_r.get_rotation_matrix_from_axis_angle([x_theta, y_theta, z_theta])
    tmp_pcd_r.rotate(R, center=(0, 0, 0))
    # Visualizing the rotated point cloud.

    draw_geoms_list = [mesh_coord_frame, tmp_pcd_r]
    o3d.visualization.draw_geometries(draw_geoms_list)

    _, pt_map = tmp_pcd_r.hidden_point_removal(camera, radius)

    # Painting all the visible points in the rotated point cloud in blue, and all the hidden points in red.

    pcd_visible = tmp_pcd_r.select_by_index(pt_map)
    pcd_visible.paint_uniform_color([0, 0, 1])    # Blue points are visible points (to be kept).
    print("No. of visible points : ", pcd_visible)

    pcd_hidden = tmp_pcd_r.select_by_index(pt_map, invert=True)
    pcd_hidden.paint_uniform_color([1, 0, 0])    # Red points are hidden points (to be removed).
    print("No. of hidden points : ", pcd_hidden)



    # Visualizing the visible (blue) and hidden (red) points in the rotated point cloud.

    draw_geoms_list = [mesh_coord_frame, pcd_visible, pcd_hidden]
    # draw_geoms_list = [mesh_coord_frame, pcd_visible]
    # draw_geoms_list = [mesh_coord_frame, pcd_hidden]

    o3d.visualization.draw_geometries(draw_geoms_list)



    # Performing the hidden point removal operation sequentially by rotating the point cloud slightly in each of the three axes
    # from -90 to +90 degrees, and aggregating the list of indexes of points that are not hidden after each operation.

    # Defining a list to store the aggregated output lists from each hidden point removal operation.
    pt_map_aggregated = []

    # Defining the steps and range of angle values by which to rotate the point cloud.
    theta_range = np.linspace(-90, 90, 7)

    # Counting the number of sequential operations.
    view_counter = 1
    total_views = theta_range.shape[0] ** 3

    # Obtaining the camera and radius parameters for the hidden point removal operation.
    camera, radius = get_hpr_camera_radius(pcd)

    # Looping through the angle values defined above for each axis.
    for x_theta_deg in theta_range:
        for y_theta_deg in theta_range:
            for z_theta_deg in theta_range:

                print(f"Removing hidden points - processing view {view_counter} of {total_views}.")

                # Rotating the point cloud by the given angle values.
                x_theta = deg2rad(x_theta_deg)
                y_theta = deg2rad(y_theta_deg)
                z_theta = deg2rad(z_theta_deg)
                pcd_rotated = get_rotated_pcd(pcd, x_theta, y_theta, z_theta)
                
                # Performing the hidden point removal operation on the rotated point cloud using the camera and radius parameters
                # defined above.
                pt_map = get_hpr_pt_map(pcd_rotated, camera, radius)
                
                # Aggregating the output list of indexes of points that are not hidden.
                pt_map_aggregated += pt_map

                view_counter += 1

    # Removing all the duplicated points from the aggregated list by converting it to a set.
    pt_map_aggregated = list(set(pt_map_aggregated))


    # Painting all the visible points in the point cloud in blue, and all the hidden points in red.

    pcd_visible = pcd.select_by_index(pt_map_aggregated)
    pcd_visible.paint_uniform_color([0, 0, 1])    # Blue points are visible points (to be kept).
    print("No. of visible points : ", pcd_visible)

    pcd_hidden = pcd.select_by_index(pt_map_aggregated, invert=True)
    pcd_hidden.paint_uniform_color([1, 0, 0])    # Red points are hidden points (to be removed).
    print("No. of hidden points : ", pcd_hidden)


    # Visualizing the visible (blue) and hidden (red) points in the point cloud.

    draw_geoms_list = [mesh_coord_frame, pcd_visible]
    # draw_geoms_list = [mesh_coord_frame, pcd_visible]
    # draw_geoms_list = [mesh_coord_frame, pcd_hidden]

    o3d.visualization.draw_geometries(draw_geoms_list)


    # Saving the point cloud with the hidden points removed as a .pcd file.

    pcd_visible_save_path = "data/3d_model_hpr.ply"
    o3d.io.write_point_cloud(pcd_visible_save_path, pcd_visible)

