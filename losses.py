import torch
from pytorch3d.ops import knn_points 
from Density_aware_Chamfer_Distance.utils_v2.model_utils import calc_dcd, calc_emd



def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  

	# Compute the nearest neighbors of the points in point_cloud_src in point_cloud_tgt
	src_nn = knn_points(point_cloud_src, point_cloud_tgt)
	tgt_nn = knn_points(point_cloud_tgt, point_cloud_src)
	
	# Get the distances of the nearest neighbors
	cham_src = src_nn.dists[..., 0]
	cham_tgt = tgt_nn.dists[..., 0]

	# Sum the distances
	loss_chamfer = cham_src.sum(1) + cham_tgt.sum(1)

	return loss_chamfer.mean()

def density_aware_chamfer_loss(point_cloud_src, point_cloud_tgt):
	loss, _, _ = calc_dcd(point_cloud_src, point_cloud_tgt)
	return loss.mean()

def earth_mover_distance(point_cloud_src, point_cloud_tgt):
	return calc_emd(point_cloud_src, point_cloud_tgt).mean()
	