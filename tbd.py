
from typing import Dict, List, Optional, Tuple

import torch
import pytorch3d
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
import os
import sys 
import utils_viz

class TBD(torch.utils.data.Dataset):
    """
    Tree Blender Dataset (TBD) or To Be Determined. TBD is a dataset of 3D trees generated from Blender.
    Args:
        data_dir: path to the TBD dataset directory, which consists of .obj and .mtl files with corresponding names.
        data_list_file: name of the file containing the list of .obj files to be loaded with the extension.
    """

    def __init__(self, data_dir, data_list_file, pc_gt_num_points=5000):
        # Dataset parameters
        self.data_dir = data_dir
        self.data_list = []
        with open(os.path.join(data_dir, data_list_file), 'r') as f:
            for line in f:
                self.data_list.append(line.strip())
        print(f"Loaded {len(self.data_list)} files from {data_list_file}.")
        self.device = torch.device("cuda:0")

        # Mesh loading parameters
        self.load_textures = True
        self.texture_resolution = 4

        # Camera rendering parameters
        self.num_views = 24
        self.elevation = 0
        self.distance = 2
        self.image_size = 256
        self._setup_renderer()
    
        # Point cloud sampling parameters
        self.pc_gt_num_points = pc_gt_num_points

    def __getitem__(self, idx):
        mesh_path = os.path.join(self.data_dir, self.data_list[idx])
        tree_dict = {}
        verts, faces, textures = self._load_mesh(mesh_path)
        mesh = self.create_mesh(verts, faces, textures)

        # Sample a random view for the image
        image_sample_idx = np.random.choice(self.num_views, 1)
        image_sample_idx = [0]
        rendered_images = self.image_from_mesh(mesh)[image_sample_idx][0][..., :3]
        R = self.R[image_sample_idx][0].to(self.device)
        T = self.T[image_sample_idx][0].to(self.device)
        gt_pointcloud = self.pointcloud_from_mesh(mesh, self.pc_gt_num_points)[0]
        # Rotate pointcloud to align with camera
        # View-aligned coordinate system
        gt_pointcloud = (R@gt_pointcloud.T).T 
        tree_dict["verts"] = verts
        tree_dict["faces"] = faces
        tree_dict["textures"] = textures
        tree_dict["mesh"] = mesh
        tree_dict["images"] = rendered_images
        tree_dict['R'] = R
        tree_dict['T'] = T
        tree_dict["pointcloud"] = gt_pointcloud
        return tree_dict

    def __len__(self):
        return len(self.data_list)

    def _load_mesh(self, model_path) -> Tuple:
        verts, faces, aux = load_obj(
            model_path,
            create_texture_atlas=self.load_textures,
            load_textures=self.load_textures,
            texture_atlas_size=self.texture_resolution,
        )
        if self.load_textures:
            textures = aux.texture_atlas
            # Some meshes don't have textures. In this case
            # create a white texture map
            if textures is None:
                textures = verts.new_ones(
                    faces.verts_idx.shape[0],
                    self.texture_resolution,
                    self.texture_resolution,
                    3,
                )
        else:
            textures = None     
        return verts, faces.verts_idx, textures

    def _setup_renderer(self):
        raster_settings = pytorch3d.renderer.RasterizationSettings(image_size=self.image_size)
        rasterizer = pytorch3d.renderer.MeshRasterizer(
            raster_settings=raster_settings,
        )
        # Shader: Given triangle, texture, lighting, etc, how should the pixel be colored?
        shader = pytorch3d.renderer.HardPhongShader(device=self.device)
        # Renderer
        self.renderer = pytorch3d.renderer.MeshRenderer(
            rasterizer=rasterizer,
            shader=shader,
        )

        # Add lights
        self.lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=self.device)

        # set 12 cameras to render images 
        self.R, self.T = pytorch3d.renderer.look_at_view_transform(
            dist=self.distance,
            elev=self.elevation,
            azim=np.linspace(0, 360, self.num_views, endpoint=False),
        )
        self.T+=torch.tensor([0, -0.7, 0])
        self.many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=self.R,
            T=self.T,
            device=self.device
        )

    def create_mesh(self, verts, faces, textures):
        verts = verts.unsqueeze(0).to(self.device)
        faces = faces.unsqueeze(0).to(self.device)
        textures = textures.unsqueeze(0).to(self.device)
        textures = pytorch3d.renderer.TexturesAtlas(textures)
        mesh = pytorch3d.structures.Meshes(verts=verts, faces=faces, textures=textures)        
        return mesh
    
    def image_from_mesh(self, mesh):
        images = self.renderer(mesh.extend(self.num_views), cameras=self.many_cameras, lights=self.lights)        
        return images
    
    def pointcloud_from_mesh(self, mesh, num_points):
        pointcloud = sample_points_from_meshes(mesh, num_points)    
        return pointcloud
    
    def debug(self, verts, faces, textures):
        verts = verts.unsqueeze(0).to(self.device)
        faces = faces.unsqueeze(0).to(self.device)
        textures = textures.unsqueeze(0).to(self.device)
        textures = pytorch3d.renderer.TexturesAtlas(textures)
        mesh = pytorch3d.structures.Meshes(verts=verts, faces=faces, textures=textures)        
        images = self.renderer(mesh.extend(self.num_views), cameras=self.many_cameras, lights=self.lights)        
        return images, mesh, self.many_cameras

def collate_batched_TBD(batch):
    if batch is None or len(batch) == 0:
        return None
    collated_dict = {}
    for k in batch[0].keys():
        collated_dict[k] = [d[k] for d in batch]
    if 'images' in collated_dict.keys():
        collated_dict['images'] = torch.stack(collated_dict['images'])    
    if 'pointcloud' in collated_dict.keys():
        collated_dict['pointcloud'] = torch.stack(collated_dict['pointcloud'])
    if 'R' in collated_dict.keys():
        collated_dict['R'] = torch.stack(collated_dict['R'])
    if 'T' in collated_dict.keys():
        collated_dict['T'] = torch.stack(collated_dict['T'])
    return collated_dict

if __name__ == "__main__":
    data_dir = os.path.join('data', 'leaves_off')
    data_list_file = 'data_list.txt'
    tbd = TBD(data_dir, data_list_file)
    for i in range(len(tbd)):
        tree_dict = tbd[i]
        verts = tree_dict["verts"]
        faces = tree_dict["faces"]
        textures = tree_dict["textures"]
        mesh = tree_dict["mesh"]
        images = tree_dict["images"]
        gt_pointcloud = tree_dict["pointcloud"]
        images, mesh, cameras = tbd.debug(verts, faces, textures)
        print(f'images.shape: {images.shape}')
        print(f'gt_pointcloud.shape: {gt_pointcloud.shape}')

        utils_viz.visualize_pointcloud(gt_pointcloud.unsqueeze(0), 256, 'pc.gif', device='cuda', export=True)

        # Show the images in a grid
        import matplotlib.pyplot as plt
        _, axs = plt.subplots(3, 8)
        axs = axs.flatten()
        for i, image in enumerate(images):
            ax = axs[i]
            ax.imshow(image.cpu())
            ax.axis("off")
        plt.show()

        # Show the mesh in interactive plotly
        from pytorch3d.vis.plotly_vis import plot_scene
        fig = plot_scene({"All Views": {"Mesh": mesh, "Cameras": cameras,},}); fig.show()

        break
    
    
