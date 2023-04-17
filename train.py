import argparse
import time
import torch
import dataset_location
import utils_viz
from losses import chamfer_loss
from tbd import TBD as TreeBlenderDataset, collate_batched_TBD
from model import SingleViewto3D

#added for debugging
import sys
import imageio
import numpy as np
import wandb

'''
python train.py --unit_test True --batch_size 1 --viz_debug True --max_iter 1000
python train.py  --batch_size 16  --max_iter 10000
'''

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    # Model parameters
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--lr', default=4e-4, type=str)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--log_freq', default=1000, type=str)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--save_freq', default=100, type=int)    
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--load_checkpoint', action='store_true')          
    parser.add_argument('--unit_test', default=False, type=bool, help='Train a single item from dataset for debugging')    
    parser.add_argument('--viz_debug', default=False, type=bool, help='visualize images, prediction GIF for debugging')    

    return parser

def train_model(args):
    tree_blender_dataset = TreeBlenderDataset(dataset_location.DATA_DIR, 
                                              dataset_location.DATA_LIST_FILE,
                                              pc_gt_num_points=args.n_points)
    loader = torch.utils.data.DataLoader(
        tree_blender_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_TBD,
        drop_last=True)
    train_loader = iter(loader)
    model =  SingleViewto3D(args)
    model.to(args.device)
    model.train()
    print(f'Train loader size: {len(train_loader)}')
    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)  # to use with ViTs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['step']
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting training !")
    read_flag = False
    loss_history = []
    prediction_img_history = []

    for step in range(start_iter, args.max_iter):

        iter_start_time = time.time()

        if step % len(train_loader) == 0: #restart after one epoch 
            train_loader = iter(loader)

        read_start_time = time.time()

        #for debugging - only load 1 item from dataset for quicker training
        if args.unit_test:
            if read_flag is False: 
                feed_dict = next(train_loader)
                read_flag = True
        else:
            feed_dict = next(train_loader)
            
        input_images = feed_dict['images']
        ground_truth_pointcloud = feed_dict['pointcloud']

        

        read_time = time.time() - read_start_time
        prediction_3d = model(input_images, args)
        loss = chamfer_loss(prediction_3d, ground_truth_pointcloud)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if args.viz_debug:
            #visualize input image, ground truth pointcloud
            utils_viz.visualize_plot_inputImg_pcloudGT(input_images, ground_truth_pointcloud)


        if (step % args.save_freq) == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'checkpoint.pth')

        if step % len(train_loader) == 0: #Save image after one epoch 
            utils_viz.visualize_pointcloud_single_image(prediction_3d, 256, 
                                            f'out/{step:05d}.png', device='cuda', export=True)
            prediction_img_history.append(imageio.imread(f'out/{step:05d}.png'))            

        #print("[%4d/%4d]; time: %.0fs (%.2f, %.2f); loss: %.3f" % (step, args.max_iter, total_time, read_time, iter_time, loss_vis))
        print(f'[{step:4d}/{args.max_iter:4d}]; ellapsed time: {total_time:.0f}s ({read_time:.2f}, {iter_time:.2f}); loss: {loss_vis:.3f}')
        wandb.log({"step": step, "loss_vis loss": loss_vis})
        
        loss_history.append(loss_vis)
    
    print('Done!')
    utils_viz.render_prediction_iterations(prediction_img_history, f'out/pcloud_iterations.gif')
    
    # Save loss to txt file
    # loss_history = np.array(loss_history)
    # np.savetxt(f'loss_{args.type}.txt', loss_history) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    wandb.init(project="L3D-final")
    train_model(args)