from torchvision import models as torchvision_models
from torchvision import transforms
import torch.nn as nn



class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # define decoder
        # Input: b x 512
        # Output: b x args.n_points x 3  
        self.n_point = args.n_points
        # Fully connected layers to output point cloud of args.n_points 
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8096),
            nn.ReLU(),
            nn.Linear(8096, self.n_point*3),
            #nn.Tanh()
        )


    def forward(self, images, args):
        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        pointclouds_pred = self.decoder(encoded_feat)
        pointclouds_pred = pointclouds_pred.reshape([-1, self.n_point, 3])
        return pointclouds_pred

