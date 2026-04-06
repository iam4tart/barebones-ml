# shared building blocks
# t-net (stn), shared mlp, global feature encoder

import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    """
    spatial transformer network (STN)
    learns a k×k transformation matrix to align input before feature extraction
    makes pointnet permutation invariant to rigid transformations (rotation, reflection)
    
    k=3 -> input transform (applied to raw xyz points)
    k=64 -> feature transform (applied to 64-dim per-point features)
    
    architecture: shared mlp per point -> max pool -> fc -> k×k matrix
    initialized to identity so training starts from no-op transform
    
    FYI: mlp is just a fully connected feedforward nn, shared mlp means
    you are applying the exact same mlp to each point in point cloud
    weights are shared across all elements
    
    shared mlp doesnt have interaction between points like CNN (k>1) where local structure matters
    """
    
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        
        # conv1d with kernel=1 is equivalent to shared fc layer per point
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        nn.init.zeros_(self.fc3.weight)
        # reshape fc3.bias 1D vector into kxk matrix and fill it with identity matrix
        nn.init.eye_(self.fc3.bias.view(k, k))
        
    def forward(self, x):
        # x: [B, k, N] — batch of point clouds
        B = x.size(0)
        
        # shared mlp per point
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # a reduction op over N elements
        # on GPU, it is parallelized accross many threads
        # collapse N points into one vector
        # this is what makes pointnet permutation invariant
        x = torch.max(x, dim=2)[0] # [B, 1024]
        
        # fc to produce k*k values
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x) # [B, k*k]
        
        # reshape to transformation matrix
        x = x.view(B, self.k, self.k)
        return x
        
def feature_transform_regularizer(trans):
    """
    trans = [B, k, k] i.e B transformation matrices one per batch sample
    
    regularization loss for the k=64-dim feature transform matrix
    encourages the learned transform to stay close to orthogonal
    orthogonal transforms preserve distances - prevents degenerate transforms
    
    loss = || I - A*A^T ||_F^2
    where A is the learned transform matrix, ||·||_F is frobenium norm
    
    without this, the 64-dim transform can collapse and hurt accuracy
    this loss is weighted at 0.001 in the total loss
    """
    B, k, _ = trans.size();
    I = torch.eye(k, device=trans.device).unsqueeze(0).expand(B, -1, -1)
    # A * A^T should be close to identity for orthogonal matrix
    diff = I - torch.bmm(trans, trans.transpose(1,2))
    # mean over batch, frobenius norm squared
    loss = torch.mean(torch.norm(diff, dim=(1,2)))
    return loss

class PointNetEncoder(nn.Module):
    """
    [B, N, 3] point cloud -> [B, 1024] global feature vector
    
    global_feat=True -> returns [B, 1024] for classification
    global_feat=False -> returns [B, 1088, N] for segmentation (concatenates global + local features per point)
    
    pipeline:
    input [B, N, 3]
    -> input t-net (3x3 transform)
    -> shared mlp (3->64)
    -> feature t-net (64x64 transform)
    -> shared mlp (64->128->1024)
    -> global max pool
    -> [B, 1024] global feature
    """
    
    def __init__(self, global_feat=True, feature_transform=True):
        super().__init__()
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.input_tnet = TNet(k=3)
        self.conv1 = nn.Conv1d(3,64,1)
        self.bn1 = nn.BatchNorm1d(64)
        
        if self.feature_transform:
            self.feature_tnet = TNet(k=64)
            
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
    def forward(self, x):
        
        B, N, _ = x.size()
        x = x.transpose(2,1)
        # learned the alignment of raw points - [B, 3, 3]
        input_trans = self.input_tnet(x)
        # apply transform to each point
        x = torch.bmm(input_trans, x)
        x = F.relu(self.bn1(self.conv1(x)))
        # learn the alignment of 64-dim features - [B, 64, 64]
        feat_trans = None
        if self.feature_transform:
            feat_trans = self.feature_tnet(x)
            x = torch.bmm(feat_trans, x)
        
        # save local features before global pool (needed for segmentation)
        point_feat = x
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # global max pool - permutation invariant aggregation
        # each output neuron is the max activation across all N points
        x = torch.max(x, dim=2)[0] # [B, 1024]
        
        if self.global_feat:
            return x, feat_trans
        
        # for segmentation: concatenate global feat with per point local feat
        x = x.unsqueeze(2).expand(-1, -1, N) # [B, 1024, N]
        x = torch.cat([x, point_feat], dim=1) # [B, 1088, N]
        return x, feat_trans