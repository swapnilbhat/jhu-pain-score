import torch
import torch.nn as nn
import torchvision.models as models

class DualXRayNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, freeze_backbone=True):
        super(DualXRayNet, self).__init__()
        
        # Load pretrained ResNet50 models for both pathways
        self.front_backbone = models.resnet50(weights='IMAGENET1K_V2')
        self.side_backbone = models.resnet50(weights='IMAGENET1K_V2')
        
        if freeze_backbone:
            # Freeze all layers except the final few
            for param in self.front_backbone.parameters():
                param.requires_grad = False
            for param in self.side_backbone.parameters():
                param.requires_grad = False
        
        # Remove the final classification layer
        feature_dim = self.front_backbone.fc.in_features
        self.front_backbone.fc = nn.Identity()
        self.side_backbone.fc = nn.Identity()
        
        # Create new layers for combined features
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, front_x, side_x):
        # Extract features from both pathways
        front_features = self.front_backbone(front_x)
#         print('fronte_features',front_features)
        side_features = self.side_backbone(side_x)
#         print('side_features',side_features)
        
        # Combine features
        combined = torch.cat((front_features, side_features), dim=1)
#         print('combined',combined)
        
        # Final prediction
        output = self.classifier(combined)
        return output