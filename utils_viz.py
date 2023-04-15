import pytorch3d
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)



def visualize_voxelgrid(voxelgrid, image_size, output_path, device='cuda', render=False, export=False, distance=1.5, start_angle=-180):
    mesh = pytorch3d.ops.cubify(voxelgrid, 0.5)
    vertices = mesh.verts_packed().unsqueeze(0)
    faces = mesh.faces_packed().unsqueeze(0)
    textures_rgb = torch.zeros_like(vertices)
    textures_rgb = textures_rgb + 0.5
    textures = pytorch3d.renderer.TexturesVertex(textures_rgb)
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)
    render_mesh(mesh, image_size, output_path, device, render, export, distance, start_angle)

def visualize_voxelgrid_single_image(voxelgrid, image_size, output_path, device='cuda', render=False, export=False, distance=1.5, start_angle=-180):
    mesh = pytorch3d.ops.cubify(voxelgrid[0], 0.5)
    vertices = mesh.verts_packed().unsqueeze(0)
    faces = mesh.faces_packed().unsqueeze(0)
    textures_rgb = torch.zeros_like(vertices)
    textures_rgb = textures_rgb + 0.5
    textures = pytorch3d.renderer.TexturesVertex(textures_rgb)
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)
    render_mesh_single_image(mesh, image_size, output_path, device, render, export, distance, start_angle)


def visualize_pointcloud(pointcloud, image_size, output_path, device='cuda', export=False):
    colors = torch.zeros_like(pointcloud)
    pc = pytorch3d.structures.Pointclouds(points=pointcloud, features=colors)
    render_pointclouds(pc, image_size, output_path, device, export)

def visualize_pointcloud_single_image(pointcloud, image_size, output_path, device='cuda', export=False):
    colors = torch.zeros_like(pointcloud)
    pc = pytorch3d.structures.Pointclouds(points=pointcloud, features=colors)
    render_pointclouds_single_image(pc, image_size, output_path, device, export)

def visualize_mesh(mesh, image_size, output_path, device='cuda', render=False, export=False):
    vertices = mesh.verts_packed().unsqueeze(0)
    faces = mesh.faces_packed().unsqueeze(0)
    textures_rgb = torch.zeros_like(vertices)
    textures_rgb = textures_rgb + 0.5
    textures = pytorch3d.renderer.TexturesVertex(textures_rgb)
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)
    render_mesh(mesh, image_size, output_path, device, render, export)

def visualize_mesh_single_image(mesh, image_size, output_path, device='cuda', render=False, export=False):
    vertices = mesh.verts_packed().unsqueeze(0)
    faces = mesh.faces_packed().unsqueeze(0)
    textures_rgb = torch.zeros_like(vertices)
    textures_rgb = textures_rgb + 0.5
    textures = pytorch3d.renderer.TexturesVertex(textures_rgb)
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)
    render_mesh_single_image(mesh, image_size, output_path, device, render, export)

def render_mesh(meshes, image_size, output_path, device, render, export, distance=1.5, start_angle=-180):
    # Rasterizer: Given a pixel, which triangles correspond to it?
    raster_settings = pytorch3d.renderer.RasterizationSettings(image_size=image_size)
    rasterizer = pytorch3d.renderer.MeshRasterizer(
        raster_settings=raster_settings,
    )
    # Shader: Given triangle, texture, lighting, etc, how should the pixel be colored?
    shader = pytorch3d.renderer.HardPhongShader(device=device)
    # Renderer
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=rasterizer,
        shader=shader,
    )

    # Add lights
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    # set 12 cameras to render images 
    num_views = 36
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=distance,
        elev=25,
        azim=np.linspace(start_angle, start_angle+360, num_views, endpoint=False),
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )

    images = renderer(meshes.extend(num_views), cameras=many_cameras, lights=lights)
    
    # Render results
    if render:
        _, axs = plt.subplots(3, 12)
        axs = axs.flatten()
        for i, image in enumerate(images):
            ax = axs[i]
            ax.imshow(image.cpu())
            ax.axis("off")
        plt.show()
        
        fig = plot_scene({
            "All Views": {
                "Mesh": meshes,
                "Cameras": many_cameras,
            },
        })
        fig.show()

    # Export GIF
    if export:
        images = images.detach().cpu().numpy()*255
        images = images.astype(np.uint8)
        imageio.mimsave(output_path, images, fps=15)

def render_mesh_single_image(meshes, image_size, output_path, device, render, export, distance=1.5, start_angle=-180):
    # Rasterizer: Given a pixel, which triangles correspond to it?
    raster_settings = pytorch3d.renderer.RasterizationSettings(image_size=image_size)
    rasterizer = pytorch3d.renderer.MeshRasterizer(
        raster_settings=raster_settings,
    )
    # Shader: Given triangle, texture, lighting, etc, how should the pixel be colored?
    shader = pytorch3d.renderer.HardPhongShader(device=device)
    # Renderer
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=rasterizer,
        shader=shader,
    )

    # Add lights
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    # set 12 cameras to render images 
    num_views = 1
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=distance,
        elev=25,
        azim=120,
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )

    images = renderer(meshes.extend(num_views), cameras=many_cameras, lights=lights)
    
    # Export GIF
    if export:
        images = images.detach().cpu().numpy()*255
        images = images.astype(np.uint8)[0]
        imageio.imwrite(output_path, images)

def render_pointclouds(pointclouds, image_size, output_path, device, export=False):
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=0.005,)
    points_renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=(1, 1, 1)),
    )

    # set 12 cameras to render images 
    num_views = 36
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=2,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    T += torch.tensor([0, -0.7, 0])
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    if export:
        # Render pc1
        rend = points_renderer(pointclouds.extend(num_views), cameras=many_cameras)[..., :3] 
        images = rend.detach().cpu().numpy()*255
        images = images.astype(np.uint8)
        print(output_path)
        print(images.shape)
        imageio.mimsave(output_path, images, fps=15)


def render_pointclouds_single_image(pointclouds, image_size, output_path, device, export=False):
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=0.005,)
    points_renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=(1, 1, 1)),
    )

    # set 12 cameras to render images 
    num_views = 1
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=2,
        elev=0,
        azim=0,
    )
    T+=torch.tensor([0, -0.7, 0])
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    if export:
        # Render pc1
        rend = points_renderer(pointclouds.extend(num_views), cameras=many_cameras)[..., :3] 
        images = rend.detach().cpu().numpy()*255
        images = images.astype(np.uint8)[0]
        imageio.imwrite(output_path, images)