import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveWeightedLoss(nn.Module):
    """
    Adaptive weighted loss that automatically adjusts weights based on training progress
    to focus more on difficult joints and time steps.
    """
    def __init__(self, n_joints, seq_len, update_interval=10, momentum=0.9):
        super().__init__()
        self.n_joints = n_joints
        self.seq_len = seq_len
        self.update_interval = update_interval  # Update weights every N batches
        self.momentum = momentum  # Momentum for weight updates
        
        # Initialize weights
        self.register_buffer('joint_weights', torch.ones(n_joints).cuda())
        # self.register_buffer('time_weights', torch.ones(seq_len).cuda())
        
        # Error history for adaptive weighting
        self.register_buffer('joint_errors', torch.zeros(n_joints).cuda())
        # self.register_buffer('time_errors', torch.zeros(seq_len).cuda())
        # self.time_errors.requires_grad = False
        # self.joint_errors.requires_grad = False
        # Batch counter
        self.batch_count = 0
    
    def forward(self, pred, target, is_training,batch_count):
        """
        Forward pass for adaptive weighted loss
        Args:
            pred: Predicted positions [batch, joints, coords, frames]
            target: Target positions [batch, joints, coords, frames]
        Returns:
            Loss value
        """
        batch_size = pred.shape[0]
        num_people = pred.shape[1]
        

        # if torch.isnan(pred).any():
        #     print("NaNs in pred!")
        # if torch.isnan(target).any():
        #     print("NaNs in target!")
        # if torch.isinf(pred).any():
        #     print("Infs in pred!")
        # if torch.isinf(target).any():
        #     print("Infs in target!")

        # print("Pred min/max:", pred.min().item(), pred.max().item())
        # print("Target min/max:", target.min().item(), target.max().item())
                # Calculate squared errors
        # pred = pred.view(batch_size,num_people,self.seq_len,self.n_joints//3, 3)
        # target = target.view(batch_size,num_people,self.seq_len,self.n_joints//3, 3)
        
        self.batch_count = batch_count+1
        squared_errors = ((pred - target) ** 2)#.sum(dim=-1)  # [batch, num_people, seq_len, joints]
            # Update error history and weights if needed
        with torch.no_grad():
            if is_training:
                
                if self.batch_count % self.update_interval == 0:
                    print(self.batch_count)
                    # Calculate errors per joint (mean across batch, coords, and frames)
                    # torch.mean(torch.norm((squared_errors), dim=3))
                    current_joint_errors = squared_errors.mean(dim=(0, 1, 2))  # [joints]
                    # current_time_errors = squared_errors.mean(dim=(0, 1, 3))  # [frames]

                    # Update joint error history with momentum
                    
                    # Calculate errors per time step (mean across batch, joints, and coords)
                    # print(current_time_errors.shape , current_time_errors.requires_grad)
                    # Update time error history with momentum
                    # with torch.no_grad():
                    self.joint_errors = self.momentum * self.joint_errors + (1 - self.momentum) * current_joint_errors
                    # self.time_errors = self.momentum * self.time_errors + (1 - self.momentum) * current_time_errors
                    
                    # Update weights (higher errors get higher weights)
                    joint_weights =  F.softmax(self.joint_errors.clone(), dim=0) * self.n_joints
                    # time_weights = F.softmax(self.time_errors, dim=0) * self.seq_len
                    
                    # Ensure weights sum to n_joints and seq_len respectively
                    self.joint_weights =  joint_weights
                    self.time_weights = time_weights

        # Apply weights
        joint_weights = self.joint_weights.view(1, 1,1,self.n_joints).expand(batch_size,num_people,self.seq_len,-1)
        #time_weights = self.time_weights.view(1, 1, self.seq_len, 1).expand(batch_size,num_people,-1,self.n_joints)
            
        # Combined weights
        weights = joint_weights #* time_weights
            
        # Weighted MSE loss
        weighted_loss = torch.mean(torch.sqrt((squared_errors * weights).sum(dim=3)))

        return weighted_loss