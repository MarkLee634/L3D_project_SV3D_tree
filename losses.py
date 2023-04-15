from pytorch3d.ops import knn_points 


def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch

	# Compute the nearest neighbors of the points in point_cloud_src in point_cloud_tgt
	src_nn = knn_points(point_cloud_src, point_cloud_tgt)
	tgt_nn = knn_points(point_cloud_tgt, point_cloud_src)
	
	# Get the distances of the nearest neighbors
	cham_src = src_nn.dists[..., 0]
	cham_tgt = tgt_nn.dists[..., 0]

	# Sum the distances
	loss_chamfer = cham_src.sum(1) + cham_tgt.sum(1)

	return loss_chamfer.mean()


