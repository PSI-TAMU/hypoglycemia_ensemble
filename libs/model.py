import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, require_bn=False, require_mp=False, require_dropout=False) -> None:
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.require_bn = require_bn
        self.require_mp = require_mp
        self.require_dropout = require_dropout
        self.relu = nn.ReLU()

        if self.require_bn:
            self.bn = nn.BatchNorm1d(out_channels)

        if self.require_mp:
            self.mp = nn.MaxPool1d(kernel_size=2, stride=2)

        if self.require_dropout:
            self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        x = self.conv(x)
        if(self.require_bn):
            x = self.bn(x)
        x = self.relu(x)
        if(self.require_mp):
            x = self.mp(x)
        if(self.require_dropout):
            x = self.dropout(x)
        return x

class ECG_CNN(nn.Module):
    def __init__(self, num_data, out_features, num_filters=50, num_layers=15, kernel_size=3, classify=True, normal_hypo_ratio=4.0, require_dropout=False) -> None:
        super().__init__()
        self.classify = classify
        self.threshold_beat = None
        self.threshold_vote = None

        layers = [conv_block(1, num_filters, kernel_size, require_bn=True, require_dropout=require_dropout)]
        for i in range(num_layers-1):
            layers.append(conv_block(num_filters, num_filters, kernel_size, require_bn=True, require_dropout=require_dropout))
        self.conv_layers = nn.Sequential(*layers)
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_data*num_filters, out_features)
        self.dropout = nn.Dropout(p=0.2)
        self.relu =  nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(out_features, 1),
        )

        self.normal_hypo_ratio = normal_hypo_ratio
        self.sigmoid = nn.Sigmoid() 
        
    def load_model(self, model_path, verbose=False):
        state_dict = torch.load(model_path)
        
        self.load_state_dict(state_dict["model_state_dict"])
        if "threshold_eer_beat" in state_dict.keys():
            self.threshold_beat = state_dict["threshold_eer_beat"]
        if "threshold_eer_vote" in state_dict.keys():
            self.threshold_vote = state_dict["threshold_eer_vote"]        

        if 'global_beat_threshold' in state_dict.keys():
            self.threshold_beat = state_dict['global_beat_threshold']
            if verbose:
                print("Global Beat Threshold: {:.3f}".format(self.threshold_beat))
        if 'global_vote_threshold' in state_dict.keys():
            self.threshold_vote = state_dict['global_vote_threshold']
            if verbose:
                print("Global Vote Threshold: {:.3f}".format(self.threshold_vote))
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.relu(x)
        if self.classify:
            x = self.classifier(x)
            x = self.sigmoid(x)
            x = x.squeeze()
        return x
        
    def loss(self, x, gt, weighted=False):
        assert x.shape == gt.shape

        if weighted:
            # Weighted BCE Loss
            pos_weight = torch.where(gt > 0.5, torch.tensor(self.normal_hypo_ratio), torch.tensor(1.0))
            return  nn.BCELoss(weight=pos_weight)(x, gt)
        else:
            return  nn.BCELoss()(x, gt)