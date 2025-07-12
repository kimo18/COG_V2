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
mode = "adaptive"
class IAFormer(nn.Module):
    def __init__(self, seq_len, d_model, opt, num_kpt, dataset, k_levels = 3 , share_d = False):
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
        if share_d:
            self.k_levels = 2

        else:
            self.k_levels = k_levels + 1
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
            for i in range(self.k_levels)])
        self.IP = IP.IP(opt=self.opt, dim_in=self.mid_feature,
                        mid_feature=self.mid_feature, num_axis=num_kpt, dropout=0.1)

        self.timecon = TimeCon.timecon_plus()

        self.AP = AP.AP(self.opt, in_features=self.opt.frame_in,
                        hidden_features=self.mid_feature, out_features = self.mid_feature)

        self.CP = CP.CP(self.opt)

        self.PE = PE.PositionalEmbedding(opt=self.opt, mid_feature=self.mid_feature, embed_size=opt.batch_size)
        # self.attention_mlp = nn.Sequential(
        #     nn.Linear(self.k_levels, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(64, self.k_levels)
        # )

        self.attention_mlp = nn.Sequential(
            nn.Linear(self.k_levels, 128),
            nn.LayerNorm(128),             # Normalizes across features
            nn.ReLU(),
            nn.Dropout(0.3),            # Helps avoid overconfidence

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, self.k_levels)         # Final logits over K levels
        )
        
        #self.linear = nn.Linear(num_kpt * self.k_levels, num_kpt)
        

    def forward(self, input_ori, gt, epoch):
        results = []
        results_feat = []
        input = torch.matmul(self.dct, input_ori)
        
        if self.dataset == "Mocap" or self.dataset == "CHI3D":
            input = input
        elif self.dataset == "Human3.6M":
            input = input.unsqueeze(dim=1)
            input_ori = input_ori.unsqueeze(dim=1)
            gt = gt.unsqueeze(dim=1)

        num_person = np.shape(input)[1]


        for i in range(num_person):
            people_in = input[:, i, :, :].clone().detach()
            if i == 0:
                people_feature_all = self.GCNQ1(people_in).unsqueeze(1).clone()
            else:
                people_feature_all = torch.cat([people_feature_all, self.GCNQ1(people_in).unsqueeze(1).clone()], 1)


        for bat_idx in range(self.opt.batch_size):
            itw = LP.Trajectory_Weight(self.opt, input_ori[bat_idx])
            if bat_idx == 0:
                bat_itw = itw.unsqueeze(0)
            else:
                bat_itw = torch.cat([bat_itw, itw.unsqueeze(0)], 0)


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
            GCN_decoder = self.GCNsQ2[0]
            feature = GCN_decoder(feature_att)
                

            if i == 0:
                level_Features = feature_att.unsqueeze(1).clone()       
                dct = feature.unsqueeze(1).clone()
                feature = torch.matmul(self.idct, feature)
                feature = feature.transpose(1, 2)
                predic = feature.unsqueeze(1).clone()
            else:
                level_Features = torch.cat([level_Features, feature_att.unsqueeze(1).clone()], 1)
                dct = torch.cat([dct, feature.unsqueeze(1).clone()], 1)
                feature = torch.matmul(self.idct, feature)
                feature = feature.transpose(1, 2)
                predic = torch.cat([predic, feature.unsqueeze(1).clone()], 1)
        results_feat.append(level_Features)        
        results.append(predic.clone())

        for k in range(1,self.k_levels):
            for bat_idx in range(self.opt.batch_size):
                itw = LP.Trajectory_Weight(self.opt, predic.transpose(2,3)[bat_idx])
                if bat_idx == 0:
                    bat_itw = itw.unsqueeze(0)
                else:
                    bat_itw = torch.cat([bat_itw, itw.unsqueeze(0)], 0)
            IP_score = self.IP(dct.clone(), (predic.clone()).transpose(2,3), num_person, bat_itw, self.AP) # input ori can be the predic from last , so we get 
            for i in range(num_person):
                people_feature = people_feature_all[:, i, :, :]#dct[:, i, :, :]
                filter_feature = people_feature.clone().detach()
                Pe = self.PE.forward(idx=i)
                feature_att = self.Att(filter_feature, IP_score, memo2=CP_ema, embedding=Pe)
                feature_att += people_feature.clone()
                GCN_decoder = self.GCNsQ2[k]
                feature = GCN_decoder(feature_att)
                if i == 0:
                    level_Features = feature_att.unsqueeze(1).clone()       
                    dct_out = feature.unsqueeze(1).clone()
                    feature = torch.matmul(self.idct, feature)
                    feature = feature.transpose(1, 2)
                    predic = feature.unsqueeze(1).clone()
                else:
                    level_Features = torch.cat([level_Features, feature_att.unsqueeze(1).clone()], 1)
                    dct_out = torch.cat([dct_out, feature.unsqueeze(1).clone()], 1)
                    feature = torch.matmul(self.idct, feature)
                    feature = feature.transpose(1, 2)
                    predic = torch.cat([predic, feature.unsqueeze(1).clone()], 1)
            dct = dct_out
            results_feat.append(level_Features)
            results.append(predic.clone())   

        # concatenated = torch.cat(results, dim=3)  # shape [B, T, D * N]
        # predic = self.linear(concatenated)
        if mode == "adaptive":
            B, P, T, J= results[0].shape
            device = results[0].device
            M = self.k_levels

            # if epoch > 30:
            filtered_results = [r[:,:,self.opt.frame_in:,:] for r in results]
            stacked_feats = torch.stack(filtered_results, dim=-1)
            gt_expanded = gt.unsqueeze(-1)  # -> (B, P, T, J, 1)
            gt_expanded = gt_expanded.transpose(2, 3)
            print(stacked_feats.shape, gt_expanded.shape)
            l2norm_error = torch.norm((stacked_feats - gt_expanded[:, :, self.opt.frame_in:, :,:]), dim=3) # -> (B, P, T, J, M)
            print(l2norm_error.shape)

            time_idx = time_idx.mean(dim=(2)) #(B,P,T,M)

            # print(time_idx.shape)
            # time_idx = torch.arange(T-self.opt.frame_in, device=device).float() / (T-self.opt.frame_in)
            # time_idx = time_idx.view(1, 1, -1)  # (1, 1, T)
            # time_idx = time_idx.expand(B, P, -1)  # (B, P, T)
            frame_joint_idx = time_idx
            selector_input = frame_joint_idx.view(-1, M)  # shape: (B*P*T, k_levels)
            print(frame_joint_idx.shape)
            # frame_joint_idx = time_idx.unsqueeze(-1)
            attn_logits = self.attention_mlp(frame_joint_idx)  #(B,P,T, M)
            attn_logits = attn_logits.view(B, P,self.opt.frame_out, M)
            def gumbel_tau_schedule(epoch, total_epochs, min_tau=0.5, max_tau=2):
                progress = epoch / total_epochs
                tau = max_tau * (1 - progress) + min_tau * progress
                return max(0.5,tau)
            # def safe_gumbel_softmax(logits, tau=1, hard=False, dim=-1, eps=1e-6):
            #     # Clamp base noise to avoid log(0)
            #     gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=eps, max=1. - eps)))
                
            #     gumbel_logits = logits + gumbel_noise
            #     return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=dim)
            tau = gumbel_tau_schedule(epoch,80)
            hard = False
            # attn_weights = safe_gumbel_softmax(attn_logits, tau=tau, hard=hard, dim=-1)
            # attn_weights = F.gumbel_softmax(attn_logits, tau=tau, hard=hard, dim=-1)  # (T, M)
            attn_weights = F.softmax(attn_logits / tau, dim=-1)
            # Stack predictions: (M, B, P, T, J)
            # filtered_results = [r[:,:,self.opt.frame_in:,:] for r in results]
            stacked = torch.stack(filtered_results, dim=0)

            # Expand attn_weights: (M, B, P, T, J)
            print(attn_weights.shape)
            attn = attn_weights.permute(3,0,1,2).unsqueeze(-1) # M,B,P,T,1
            print(attn.shape)

            # attn = attn_weights.permute(2, 0, 1).unsqueeze(1).unsqueeze(1)  # (M, 1, 1, T, J)

            # Multiply and sum: final shape (B, P, T, J)
            fused = (attn * stacked).sum(dim=0)
            # print(fused.shape)
            predic = torch.cat([results[0][:,:,:self.opt.frame_in,:], fused], 2)

            with torch.no_grad():
                # probs = F.softmax(attn_logits, dim=-1)
                max_idx = attn_weights.argmax(dim=-1)  # shows which level was chosen
                print("Selected levels (frame, joint):", max_idx)  # shape: (T, J)
                print("this is the mean of each level",max_idx.float().mean())
                # print("True selected ", attn_weights)

            # avg_weights = attn_weights.mean(dim=0)  # (K,)
            # uniform = torch.full_like(attn_weights, 1.0 / attn_weights.shape[-1])
            # hard_target = F.one_hot(attn_weights.argmax(dim=-1), num_classes=attn_weights.shape[-1]).float()
            # def kl_alpha_schedule(epoch, total_epochs):
            #     return epoch / total_epochs  # goes from 0 (uniform) to 1 (one-hot)
            # if epoch <= 40:
            #     alpha = kl_alpha_schedule(epoch, 40)
            #     dynamic_target = (1 - alpha) * uniform + alpha * hard_target
            # else:
            #     dynamic_target = hard_target
            # entropy = -torch.sum(attn_weights * attn_weights.clamp(min=1e-8).log(), dim=-1)
            # entropy_loss = entropy.mean()
            # KL-divergence to push distribution toward uniform usage
            # kl_loss = F.kl_div(attn_weights.log(), uniform, reduction='batchmean')
            # entropy = -(attn_weights * attn_weights.log()).sum(dim=-1).mean()
            # Uniform target
            # uniform_target = torch.full_like(attn_weights, 1.0 / attn_weights.shape[-1])
            # # Annealed weight
            # alpha = max(0.0, 1.0 - epoch / (0.4 * 80))  # Anneal over 60% of training
            # kl_loss = F.kl_div(attn_weights.log(), uniform_target, reduction='batchmean')
            # # loss += alpha * kl_loss  # Weight decreases as training progresses
            # loss = self.mix_loss(predic, gt) + self.w_cb * k_loss  + alpha * kl_loss#(0.01*entropy)#(0.1*entropy_loss)#
            entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean()  # scalar
            loss = self.mix_loss(predic, gt) + self.w_cb * k_loss #+ 0.001 * entropy

            # #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            # else:
        if mode == "static":
            predic = torch.cat([results[0][:,:,:self.opt.frame_in+15,:],results[-1][:,:,self.opt.frame_in+15:,:]], 2)
            #     for p in self.attention_mlp.parameters():
            #         p.requires_grad = False
            loss = self.mix_loss(predic, gt) + self.w_cb * k_loss 



        # predic = torch.cat([results[0][:,:,:self.opt.frame_in,:], fused], 2)
        

        if self.dataset == "Mocap" or self.dataset == "CHI3D" or self.dataset == "Wusi":
            return predic, loss
        elif self.dataset == "Human3.6M":
            return predic[:, 0, :, :], loss

    def mix_loss(self, predic, gt):

        gt = gt.transpose(2, 3)
        bs, np, sql, _ = gt.shape

        spacial_loss_pred = torch.mean(torch.norm((predic[:, :, self.opt.frame_in:, :] - gt[:, :, self.opt.frame_in:, :]), dim=3))
        spacial_loss_ori = torch.mean(torch.norm((predic[:, :, :self.opt.frame_in, :] - gt[:, :, :self.opt.frame_in, :]), dim=3))
        spacial_loss = spacial_loss_pred + spacial_loss_ori * 0.1

        temporal_loss = 0


        for idx_person in range(np):

            
            tempo_pre = self.timecon(predic[:, idx_person, :, :].unsqueeze(1))
            tempo_ref = self.timecon(gt[:, idx_person, :, :].unsqueeze(1))
            
            temporal_loss += torch.mean(torch.norm(tempo_pre-tempo_ref, dim=3))

        loss = self.w_sp * spacial_loss + self.w_tp * temporal_loss

        return loss


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