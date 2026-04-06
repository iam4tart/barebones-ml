import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_regularizer

class PointNetCls(nn.Module):
    """
    [B, N, 3] point cloud -> [B, num_classes] logits
    """
    
    def __init__(self, num_classes=40, feature_transform=True):
        super().__init__()
        self.feature_transform = feature_transform
        self.encoder = PointNetEncoder(global_feat=True, feature_transform=feature_transform)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        # prevents overfitting - pointnet uses 0.3
        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.3)
        
    def forward(self, x):
        global_feat, feat_trans = self.encoder(x)
        
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        
        return x, feat_trans
    
    def get_loss(self, logits, labels, feat_trans, reg_weight=0.001):
        """
        total loss = cross entropy + feature transform regularization
        reg_weight = 0.001 matches original pointnet paper
        """
        ce_loss = F.cross_entropy(logits, labels)
        reg_loss = 0.0
        
        if self.feature_transform and feat_trans is not None:
            reg_loss = feature_transform_regularizer(feat_trans)
            
        return ce_loss + reg_weight*reg_loss, ce_loss, reg_loss