import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from model.layers import TimeCon
from model.layers import IPM as IP
from model.layers import Attention as Att
from model.layers import GCN

from model.layers import IPM_IAW as AP
from model.layers import IPM_ITW as LP
from model.layers import IPLM as CP

from model.layers import PE
from utils import loss_functions as WL


kofta = False

class PosePredictionModel(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2, seq_len=75, short_term_window=15, frame_in=25):
        super().__init__()
        self.short_term_window = short_term_window
        self.frame_in = frame_in
        # Short-Term Transformer (focusing on last 15 frames)
        self.short_term_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads), num_layers=num_layers // 2
        )
        
        # Long-Term Transformer (full sequence)
        self.long_term_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads), num_layers=num_layers
        )
        self.expand_layer_short = nn.Linear(short_term_window, embed_dim)
        self.expand_layer_long = nn.Linear(frame_in, embed_dim)
        self.dimense_layer_short = nn.Linear(embed_dim, short_term_window)
        self.dimense_layer_long = nn.Linear(embed_dim, seq_len-frame_in-short_term_window)
        # Adaptive fusion weights
        self.short_term_weight = nn.Parameter(torch.tensor(0.5))  # Learnable blending factor
        self.long_term_weight = nn.Parameter(torch.tensor(0.5))
        
        self.output_layer = nn.Linear(embed_dim, 45)  # Output 3D pose

            
    def forward(self, x):
        """
        x: Input shape (batch, num_people, seq_len, joints) -> (96, 5, 75, 45)
        """
        batch_size, num_people, joint_dim ,seq_len= x.shape
        # x = x.view(batch_size * num_people, seq_len, joint_dim)  # Flatten people dimension

        # Short-term processing (last 15 frames)
        refined_output = torch.empty(batch_size,num_people,joint_dim,seq_len)
        for i in range(batch_size):
            input_new = x[i,:,:,:].clone().detach()
            short_term_x = input_new[:, :, self.frame_in-self.short_term_window:self.frame_in].clone().detach()
            long_term_x = input_new[:,:,:self.frame_in].clone().detach()

            short_term_x = self.expand_layer_short(short_term_x)
            long_term_x = self.expand_layer_long(long_term_x)

            # print(short_term_x.shape)
            short_term_features = self.short_term_transformer(short_term_x)

            # Long-term processing (entire sequence)
            long_term_features = self.long_term_transformer(long_term_x)
            
            short_term_features = self.dimense_layer_short(short_term_features)
            long_term_features = self.dimense_layer_long(long_term_features)

            # Upsample short-term features to match full sequence size
            # short_term_features = torch.nn.functional.interpolate(short_term_features.permute(0, 2, 1), size=seq_len).permute(0, 2, 1)

            # Fuse both outputs using learnable weights
            short_term_output = x[i,:,:,self.frame_in:self.frame_in+self.short_term_window]+ self.short_term_weight * short_term_features
            long_term_output  = x[i,:,:,self.frame_in+self.short_term_window:] + self.long_term_weight * long_term_features
            # Final pose prediction
            output = torch.cat([x[i,:,:,:self.frame_in],short_term_output,long_term_output], 2)
            if i == 0:
                refined_output = output.unsqueeze(0).clone()
            else:
                refined_output = torch.cat([refined_output, output.unsqueeze(0).clone()], 0)

            # output = self.output_layer(fused_features)
        return refined_output  # Restore original shape

class TemporalAttentionRefinement(nn.Module):
    def __init__(self, d_model=64, num_heads=4, dropout=0.1):
        super(TemporalAttentionRefinement, self).__init__()
        # Temporal self-attention
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # MLP-based feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Residual connection for refinement
        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, pred):
        """
        pred: Tensor of shape [batch_size, num_person, seq_len, num_kpt]
        """

        # print(pred.shape)
        bs,num_person,num_kpt,seq_len = pred.shape
        # pred = pred.transpose(2, 3)
        refined_output = torch.empty(bs,num_person,seq_len, num_kpt)
        for i in range(bs):
            persons = pred[i,:,:,:].clone().detach() 
        # print(bs, num_person, seq_len, num_kpt)
        # Reshape to [batch_size * num_person, seq_len, num_kpt]
        

        # Apply Layer Norm before attention
            persons = self.norm1(persons)

            # Temporal attention
            refined, _ = self.attention(persons, persons, persons)

            # Residual Connection + MLP Transformation
            refined = refined + self.mlp(self.norm2(refined))

            # Residual weight scaling
            refined = pred[i,:,:,:]+ self.residual_weight * refined
            
            if i == 0:
                refined_output = refined.unsqueeze(0).clone()
            else:
                refined_output = torch.cat([refined_output, refined.unsqueeze(0).clone()], 0)
            # Reshape back to original shape
            # refined = refined.transpose(2, 3)

        return refined_output




class IAFormer(nn.Module):
    def __init__(self, seq_len,d_model, opt, num_kpt,dataset,
            k_levels=3, share_d=False):
        super(IAFormer, self).__init__()
        self.opt = opt
        self.mid_feature = opt.seq_len
        self.dataset = dataset
        self.seq_len = seq_len
        self.num_kpt = num_kpt
        self.dct, self.idct = self.get_dct_matrix(self.num_kpt)

        if self.opt.exp == 'Mocap':
            self.w_sp = self.opt.w_sp
            self.w_tp = self.opt.w_tp
            self.w_cb = self.opt.w_cb
        else:
            self.w_sp = 1
            self.w_tp = 1
            self.w_cb = 1
        if share_d:
            depth = 2

        else:
            depth = k_levels + 1
        
        self.Att = Att.TransformerDecoder(num_layers=self.opt.num_layers,
                                          num_heads=5,
                                          d_model=self.mid_feature,
                                          d_ff=self.mid_feature,
                                          dropout=0.1)  # 6
    

        self.GCNQ1 = GCN.GCN(input_feature=self.mid_feature,
                             hidden_feature=d_model,
                             p_dropout=0.3,
                             num_stage=self.opt.num_stage,
                             node_n=num_kpt)#2

        # self.GCNQ2 = GCN.GCN(input_feature=self.mid_feature,
        #                      hidden_feature=d_model,
        #                      p_dropout=0.3,
        #                      num_stage=self.opt.num_stage,
        #                      node_n=num_kpt)

        self.GCNsQ2 = nn.ModuleList([
           GCN.GCN(input_feature=self.mid_feature,
                             hidden_feature=d_model,
                             p_dropout=0.3,
                             num_stage=self.opt.num_stage,
                             node_n=num_kpt)
            for i in range(depth)])

        self.IP = IP.IP(opt=self.opt, dim_in=self.mid_feature,
                        mid_feature=self.mid_feature, num_axis=num_kpt, dropout=0.1)

        self.timecon = TimeCon.timecon_plus()

        self.AP = AP.AP(self.opt, in_features=self.opt.frame_in,
                        hidden_features=self.mid_feature, out_features = self.mid_feature)

        self.CP = CP.CP(self.opt)

        self.PE = PE.PositionalEmbedding(opt=self.opt, mid_feature=self.mid_feature, embed_size=opt.batch_size)
        #self.refinement = TemporalAttentionRefinement(d_model=opt.seq_len, num_heads=2, dropout=0.1)
        #self.refinement_V2 = PosePredictionModel(seq_len=opt.seq_len,short_term_window=15,frame_in=opt.frame_in)
        self.k_levels = k_levels


        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(delta=0.1)  # More robust to outliers
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.WL = WL.AdaptiveWeightedLoss(self.num_kpt, self.seq_len-self.opt.frame_in)
        # self.pooling = nn.AdaptiveAvgPool1d(25)
        # self.final_projection = nn.Linear(opt.seq_len, opt.frame_out)
        # if share_d:
        #     depth = 2

        # else:
        #     depth = k_levels + 1
    def root_relative_joints(self,input_ori):
        # print(input_ori.shape)
        bs,n_people,joints,seq_len = input_ori.shape
        input_ori = input_ori.view(bs,n_people,joints//3,3,seq_len)
        root_joint = input_ori[:, :, :1, :, :]  # Extract root joint
        relative_joints = input_ori - root_joint  # Normalize all joints relative to root
        input_ori = input_ori.view(bs,n_people,joints,seq_len)
        relative_joints = relative_joints.view(bs,n_people,joints,seq_len)
        return relative_joints, root_joint
    
    def forward(self, input_ori, gt, batch_index):

        # input_ori = input_ori.permute(0, 1, 3, 2)
        bs,n_people,joints,seq_len = input_ori.shape
        if kofta :
            relative_joints, root_joint = self.root_relative_joints(input_ori)
            input = torch.matmul(self.dct, relative_joints)
        else:
            input = torch.matmul(self.dct, input_ori)

        if self.dataset == "Mocap" or self.dataset == "CHI3D" or self.dataset == "Wusi":
            input = input
        elif self.dataset == "Human3.6M": # lets account for the 1 person in the scene and have same tensor shap
            input = input.unsqueeze(dim=1)
            input_ori = input_ori.unsqueeze(dim=1)
            if kofta:
                relative_joints = relative_joints.unsqueeze(dim=1)
            root_joint = root_joint.unsqueeze(dim=1)
            gt = gt.unsqueeze(dim=1)

        num_person = np.shape(input)[1]
        for i in range(num_person):

            people_in = input[:, i, :, :].clone().detach()
            # print(people_in.shape)
            if i == 0:
                people_feature_all = self.GCNQ1(people_in).unsqueeze(1).clone()
            else:
                people_feature_all = torch.cat([people_feature_all, self.GCNQ1(people_in).unsqueeze(1).clone()], 1)
        # people_local_feature_all = people_feature_all.clone()
        for k in range(self.k_levels+1):
            if k > 0:
                print("hi_2")
                if kofta:
                    input = relative_joints
                for i in range(num_person):
                    people_in = input[:, i, :, :].clone()#.detach()
                    if i == 0:
                        people_feature_all = self.GCNQ1(people_in).unsqueeze(1).clone()
                    else:
                        people_feature_all = torch.cat([people_feature_all, self.GCNQ1(people_in).unsqueeze(1).clone()], 1)
            # if k == 0:
            #     people_local_feature_all = people_feature_all.clone()


            # print(f"this is level {k} and this is the  people feature {people_feature_all.shape}")
            for bat_idx in range(self.opt.batch_size):
                itw = LP.Trajectory_Weight(self.opt, input_ori[bat_idx])
                if bat_idx == 0:
                    bat_itw = itw.unsqueeze(0)
                else:
                    bat_itw = torch.cat([bat_itw, itw.unsqueeze(0)], 0)

            # print(relative_joints.shape)
            IP_score = self.IP(people_feature_all.clone(), input_ori.clone(), num_person, bat_itw, self.AP)

            CP_score, k_loss, CP_ema = self.CP(IP_score.clone())
            IP_feature = IP_score
            CP_ema = CP_ema * CP_score


            for i in range(num_person):

                people_feature = people_feature_all[:, i, :, :]
                
                filter_feature = people_feature.clone().detach()

                Pe = self.PE.forward(idx=i)
                feature_att = self.Att(filter_feature, IP_feature, memo2=CP_ema, embedding=Pe)
                feature_att += people_feature.clone()
                GCNQ2 = self.GCNsQ2[k]
                feature = GCNQ2(feature_att)
                feature = torch.matmul(self.idct, feature)
                feature = feature.transpose(1, 2)
                # print("this  is the shape of feature",feature.shape)

                # feature = self.GCNQ2(feature_att)
                #feature = self.pooling(feature) 
                # feature = self.final_projection(feature)
                # if k == self.k_levels :
                #     refined_pred = self.refinement(feature)
               

                # print(feature.shape)    
                if i == 0:
                    #idcted = torch.matmul(self.idct, feature).unsqueeze(1)
                    predic = feature.unsqueeze(1).clone()
                    
                else:
                    #idcted = torch.cat([idcted, torch.matmul(self.idct, feature).unsqueeze(1)], 1)
                    predic = torch.cat([predic, feature.unsqueeze(1).clone()], 1)
            # people_feature_all = predic.clone().detach()
            # people_feature_all = predic.clone().detach()

            if k < self.k_levels:
                print("hi")
                # input = predic
                if kofta :
                    relative_joints = predic.clone()
                    input_ori = idcted.clone().view(bs,n_people,joints//3,3,seq_len)
                    input_ori += root_joint
                    input_ori = input_ori.view(bs,n_people,joints,seq_len)
                else:
                    input = predic.clone()
                    input_ori = idcted.clone()
        # print("hi",predic.shape)
        # print(f"this is the predic shape {predic.shape}")
        # print(f"this is the gt shape {gt.shape}")

        # print(predic.shape)
        # predic = torch.matmul(self.idct, predic)
        if kofta:
            idcted = idcted.view(bs,n_people,joints//3,3,seq_len)
            idcted += root_joint
            idcted = idcted.view(bs,n_people,joints,seq_len)

        # predic = predic.view(ba)
        # predic = self.refinement(predic)
        # predic = self.refinement_V2(predic)
        # print(predic.shape)
        #idcted = idcted.transpose(2, 3)
       


        loss = self.mix_loss(predic, gt,batch_index) + self.w_cb * k_loss

        # predic = predic[:,:,1:,:]
        # print(predic.shape)
        if self.dataset == "Mocap" or self.dataset == "CHI3D" or self.dataset == "Wusi":
            return predic, loss
        elif self.dataset == "Human3.6M":
            return predic[:, 0, :, :], loss

    def mix_loss(self, predic, gt,batch_index):

        gt = gt.transpose(2, 3)
        bs, np, sql, _ = gt.shape
        # spacial_loss_pred = torch.mean(((gt[:, :, self.opt.frame_in:, :]-predic[:, :, self.opt.frame_in:, :])**2).sum(dim=-1))
        # torch.mean(torch.norm((predic - gt), dim=3)) 
        
        

        #spacial_loss_pred = torch.mean((torch.norm((predic[:, :, 1:, :] - gt[:, :, self.opt.frame_in:, :]))))
        spacial_loss_pred = torch.mean(torch.norm((predic[:, :, self.opt.frame_in:, :] - gt[:, :, self.opt.frame_in:, :]), dim=3))
        
        # recon_loss = nn.functional.l1_loss(predic[:, :,0,:], gt[:, :, 49, :])
        #torch.mean(((gt[:, :, :self.opt.frame_in, :]-predic[:, :, :self.opt.frame_in, :])**2).sum(dim=-1))
        spacial_loss_ori =  torch.mean(torch.norm((predic[:, :, :self.opt.frame_in, :] - gt[:, :, :self.opt.frame_in, :]), dim=3))
        
        # if batch_index >= 0:
        #     WL_output=self.WL(predic[:, :, self.opt.frame_in:, :] , gt[:, :, self.opt.frame_in:, :],True,batch_index)
        # else:
        #     WL_output = spacial_loss_pred   
        # physics_loss_pred = self.physics_loss(predic, gt)


        # spacial_loss = WL_output + (spacial_loss_ori * 0.1) +(physics_loss_pred *0.5)
        # print(physics_loss_pred,WL_output,spacial_loss_ori,spacial_loss)

        # short_long_loss = self.short_long_loss(predic,gt,15)
        spacial_loss =  (spacial_loss_pred) + (spacial_loss_ori * 0.1)# +(physics_loss_pred*0.5) #recon_loss* 0.1 

        temporal_loss = 0


        for idx_person in range(np):

            
            tempo_pre = self.timecon(predic[:, idx_person, :, :].unsqueeze(1))
            tempo_ref = self.timecon(gt[:, idx_person, :, :].unsqueeze(1))
            
            temporal_loss += torch.mean(((tempo_ref-tempo_pre)**2).sum(dim=-1))

        loss = self.w_sp * spacial_loss + self.w_tp * temporal_loss

        return loss


    def physics_loss(self,pred, gt):
        """Enforce smooth movement by constraining velocity and acceleration."""
        velocity_pred = pred[:,:, 1:,:] - pred[:,:, :-1,:]  # Velocity = ΔPosition
        velocity_gt = gt[:,:, 1:,:] - gt[:,:, :-1,:]
        acceleration_pred = velocity_pred[:,:, 1:,:] - velocity_pred[:,:, :-1,:]  # Acceleration = ΔVelocity
        acceleration_gt = velocity_gt[:,:, 1:,:] - velocity_gt[:,:, :-1,:]

        velocity_loss = torch.mean((velocity_pred - velocity_gt) ** 2)
        acceleration_loss = torch.mean((acceleration_pred - acceleration_gt) ** 2)

        return velocity_loss #+ 0.1 * acceleration_loss  # Weight acceleration loss less to avoid oversmoothing
    
    def short_long_loss(self, pred, gt,short_term_window):

        short_term_pred = pred[:, :, self.opt.frame_in:self.opt.frame_in+short_term_window, :]
        long_term_pred = pred[:, :, self.opt.frame_in+short_term_window:, :]
        short_term_target = gt[:, :, self.opt.frame_in:self.opt.frame_in+short_term_window, :]
        long_term_target = gt[:, :, self.opt.frame_in+short_term_window:, :]

        # Short-Term: Use stricter loss (Huber + MSE)
        # short_term_loss = self.huber_loss(short_term_pred, short_term_target) + self.mse_loss(short_term_pred, short_term_target)
        short_term_loss =  torch.mean(torch.norm((short_term_pred - short_term_target), dim=3))
        # Long-Term: Focus on smooth transitions (Smooth L1)
        # long_term_loss = self.smooth_l1_loss(long_term_pred, long_term_target)
        long_term_loss = torch.mean(torch.norm((long_term_pred - long_term_target), dim=3))
        # Weighted sum (adjust weights based on training needs)
        total_loss = 1 * short_term_loss + 0.9 * long_term_loss
        return total_loss

    def get_dct_matrix(self, N):
        # Computes the discrete cosine transform (DCT) matrix and its inverse (IDCT)
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        dct_m = torch.tensor(dct_m).float().cuda()
        idct_m = torch.tensor(idct_m).float().cuda()
        return dct_m, idct_m



class mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x