import argparse
import time
import torch
from model import SingleViewto3D
from tbd import TBD as TreeBlenderDataset, collate_batched_TBD
from model import SingleViewto3D
import utils_viz


import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import matplotlib.pyplot as plt 

#added for debugging
import sys

'''

python eval_model.py --type 'point' --load_checkpoint  
python eval_model.py --type 'mesh' --load_checkpoint   --unit_test True 

'''

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=10000, type=str)
    parser.add_argument('--vis_freq', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--unit_test', default=False, type=bool)
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)

    gt_points = gt_points.to(args.device)
    pred_points = pred_points.to(args.device)
    
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics


def evaluate_model(args):

    tree_blender_dataset = TreeBlenderDataset(dataset_location.DATA_DIR, 
                                              dataset_location.TEST_LIST_FILE,
                                              pc_gt_num_points=args.n_points)
    loader = torch.utils.data.DataLoader(
        tree_blender_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_TBD,
        drop_last=True)

    eval_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        # checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        checkpoint = torch.load(f'checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
   
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    read_flag = False

    print(f" start iter and max iter : {start_iter} {max_iter}")
    for step in range(start_iter, max_iter):

        iter_start_time = time.time()
        read_start_time = time.time()

        if args.unit_test:
            if read_flag is False: 
                feed_dict = next(eval_loader)
                read_flag = True
        else:
            try:
                feed_dict = next(eval_loader)
            except StopIteration:
                print("stop iteration")
                eval_loader = iter(loader)
                feed_dict = next(eval_loader)

        #get data
        images_gt, mesh_gt = preprocess(feed_dict, args)

        mesh_gt = mesh_gt[0]
        # utils_viz.visualize_plot_inputImg_meshGT(images_gt, mesh_gt)
        

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        if args.type == "vox":
            predictions = predictions.permute(0,1,4,3,2)

        predictions = predictions.to(args.device)
        mesh_gt = mesh_gt.to(args.device)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)



        # TODO:
        if (step % args.vis_freq) == 0:
            # visualization block
            # render image from mesh

            if args.type == "vox":
                print(f" shape of pred points : {predictions.shape}")
                # rendered_gt, rendered_pred = utils_vox.get_both_renders_pred_gt(mesh_gt, predictions, args)
            
            elif args.type == "point":

                rendered_gt = utils_viz.get_image_from_mesh(mesh_gt, 256, args.device)
                rendered_pred = utils_viz.get_image_from_pointcloud(predictions, 256, args.device)

            elif args.type == "mesh":
                print(f" shape of pred points : {predictions.shape}")
                # rendered_gt = utils_vox.render_from_mesh_by_adding_texture(mesh_gt, args.device)
                # rendered_pred = utils_vox.render_from_mesh_by_adding_texture(predictions, args.device)


            # plt.imsave(f'vis/{step}_{args.type}.png', rend)
            # plt.figure()

            # subplot(r,c) provide the no. of rows and columns
            f, axarr = plt.subplots(1,3)

            # use the created array to output your multiple images. In this case I have stacked 4 images vertically
            images_gt_ = images_gt.detach().cpu().numpy()

            images_gt_ = images_gt_.squeeze(0)
            axarr[0].imshow(images_gt_)
            axarr[0].set_title('Input Image')
            axarr[1].imshow(rendered_pred)
            axarr[1].set_title('Pred pcloud')
            axarr[2].imshow(rendered_gt)
            axarr[2].set_title('GT mesh')
            plt.show()

            

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
