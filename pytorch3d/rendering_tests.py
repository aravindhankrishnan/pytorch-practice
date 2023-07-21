import argparse
from fisheyecameras import FishEyeCameras
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch3d
import torch
import torch.nn.functional as F

from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
        look_at_view_transform,
        FoVOrthographicCameras, 
        FoVPerspectiveCameras,
        PerspectiveCameras,
        PointsRasterizationSettings,
        PointsRenderer,
        PulsarPointsRenderer,
        PointsRasterizer,
        AlphaCompositor,
        NormWeightedCompositor
        )

def Rx(theta):
      return np.array([[ 1, 0           , 0           ],
                       [ 0, math.cos(theta),-math.sin(theta)],
                       [ 0, math.sin(theta), math.cos(theta)]])
        
def Ry(theta):
    return np.array([[ math.cos(theta), 0, math.sin(theta)],
                     [ 0           , 1, 0           ],
                     [-math.sin(theta), 0, math.cos(theta)]])
              
def Rz(theta):
    return np.array([[ math.cos(theta), -math.sin(theta), 0 ],
                     [ math.sin(theta), math.cos(theta) , 0 ],
                     [ 0           , 0            , 1 ]])

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device

def load_point_cloud(filename, device):
    pointcloud = np.load(filename)
    verts = torch.Tensor(pointcloud['verts']).to(device)
    rgb = torch.Tensor(pointcloud['rgb']).to(device)
    point_cloud = Pointclouds(points=[verts], features=[rgb])
    return point_cloud

FOCAL_LENGTH = 80.56
CX = 112.77
CY = 40.77
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 131

def get_focal_length():
    return torch.tensor([[FOCAL_LENGTH]])

def get_principal_point():
    return torch.tensor([[CX, CY]])

def get_radial_distortion():
    return torch.tensor([[0., 0., 0., 0., 0., 0.]])

def get_tangential_distortion():
    return torch.tensor([[0., 0.]])

def get_thin_prism_distortion():
    return torch.tensor([[0., 0., 0., 0]])

def get_image_size():
    return torch.tensor([[IMAGE_HEIGHT, IMAGE_WIDTH]])

def get_K_matrix():
    return torch.tensor([[ [FOCAL_LENGTH,  0,    CX,   0.],
                          [0.,    FOCAL_LENGTH, CY,    0.],
                          [0.,     0.,     0.,     1.],
                          [0.,     0.,     1.,    0.]] ])


def get_orthographic_cameras(device):
    return FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

def get_fisheye_cameras(R, T, device):
    focal_length = get_focal_length()
    principal_point = get_principal_point()
    radial_distortion = get_radial_distortion()    
    tangential_distortion = get_tangential_distortion()    
    thin_prism_distortion = get_thin_prism_distortion()    
    image_size = get_image_size()

    K = get_K_matrix()
    print('K matrix', K)
    return FishEyeCameras(K,
                          focal_length,
                          principal_point,
                          radial_distortion,
                          tangential_distortion,
                          thin_prism_distortion, 
                          R,
                          T,
                          world_coordinates=False,
                          use_radial=False,
                          use_tangential=False,
                          use_thin_prism=False,
                          device=device,
                          in_ndc=False,
                          image_size=image_size)

def get_perspective_cameras(R, T, device):
    K = get_K_matrix()
    return PerspectiveCameras(R=R, T=T, K=K, in_ndc=False, image_size=get_image_size(), device=device)

def get_fov_perspective_cameras_with_K(R, T, device):
    K = get_K_matrix()
    return FoVPerspectiveCameras(R=R, T=T, K=K, device=device)

def get_fov_perspective_cameras_with_fov(R, T, device):    
    return FoVPerspectiveCameras(znear=0.01, zfar=10.0, aspect_ratio=1.0, fov=110, degrees=True, R=R, T=T, device=device)

def main():
    parser = argparse.ArgumentParser(description='testing rendering')
    parser.add_argument('--point_cloud', type=str, help='Input point cloud', required=True)
    parser.add_argument('--camera_type', type=str, help='camera type ', choices=['perspective', 'fisheye'], required=True)
    args = parser.parse_args()

    device = setup_device()

    point_cloud = load_point_cloud(args.point_cloud, device)

    rx, ry, rz  = math.radians(0), math.radians(0), math.radians(0)
    R = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([[rx, ry, rz]]), 'XYZ')
    T = torch.tensor([[0., 0., 0.]])

    if args.camera_type == 'fisheye':
        cameras = get_fisheye_cameras(R, T, device)
    elif args.camera_type == 'perspective':
        cameras = get_perspective_cameras(R, T, device)

    #cameras = get_orthographic_cameras(device)
    #cameras = get_fov_perspective_cameras_with_K(R, T, device)
    #cameras = get_fov_perspective_cameras_with_fov(R, T, device)
    #cameras = get_perspective_cameras(R, T, device)

    # print(cameras.K)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters. 
    raster_settings = PointsRasterizationSettings(image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), radius = 0.003, points_per_pixel = 10)
    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
    images = renderer(point_cloud)
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.savefig(os.path.expanduser('~/image.png'))

    # plt.imshow(images[0, ..., :3].cpu().numpy())

if __name__ == '__main__':
    main()
