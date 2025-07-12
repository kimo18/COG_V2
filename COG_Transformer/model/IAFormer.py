import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from model.layers import TimeCon
from model.layers import IPM as IP
from model.layers import Attention as Att
from model.layers import GCN

from model.layers import IPM_IAW as AP
from model.layers import IPM_ITW as LP
from model.layers import IPLM as CP

from model.layers import PE
from utils import discriminator_loss 
from utils import feats_extractor

kofta = False
mode = "kofta"



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
        

    def forward(self, input_ori, gt, epoch , Train= True , FineTune = True):
        if Train:
            training = True
        else:
            training = False    
        results = []
        people_feat_results = []
        people_ip_results = []
        people_att = []
        input = torch.matmul(self.dct, input_ori)
        print("what is thr ehape 111",input_ori.shape)
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


        IP_score, ip_each = self.IP(people_feature_all.clone(), input_ori.clone(), num_person, bat_itw, self.AP)

        CP_score, k_loss, CP_ema = self.CP(IP_score.clone())
        IP_feature = IP_score
        CP_ema = CP_ema * CP_score


        for i in range(num_person):

            people_feature = people_feature_all[:, i, :, :]
            
            filter_feature = people_feature.clone().detach()

            Pe = self.PE.forward(idx=i)
            feature_att = self.Att(filter_feature, IP_feature, memo2=CP_ema, embedding=Pe)
            feature_att += people_feature.clone()      
            if i==0:
               feature_att_all  =  feature_att.unsqueeze(1) 
            else:
                feature_att_all  =  torch.cat([feature_att_all,feature_att.unsqueeze(1)],1)      
            GCN_decoder = self.GCNsQ2[0]
            feature = GCN_decoder(feature_att)
                

            if i == 0:
                # level_Features = feature_att.unsqueeze(1).clone()       
                dct = feature.unsqueeze(1).clone()
                feature = torch.matmul(self.idct, feature)
                feature = feature.transpose(1, 2)
                predic = feature.unsqueeze(1).clone()
            else:
                # level_Features = torch.cat([level_Features, feature_att.unsqueeze(1).clone()], 1)
                dct = torch.cat([dct, feature.unsqueeze(1).clone()], 1)
                feature = torch.matmul(self.idct, feature)
                feature = feature.transpose(1, 2)
                predic = torch.cat([predic, feature.unsqueeze(1).clone()], 1)
        # results_feat.append(level_Features)
        people_feat_results.append(people_feature_all)
        people_feat_results.append(dct)
        people_att.append(feature_att_all)
        people_ip_results.append(ip_each)        
        results.append(predic.clone())

        for k in range(1,self.k_levels):
            for bat_idx in range(self.opt.batch_size):
                itw = LP.Trajectory_Weight(self.opt, predic.transpose(2,3)[bat_idx])
                if bat_idx == 0:
                    bat_itw = itw.unsqueeze(0)
                else:
                    bat_itw = torch.cat([bat_itw, itw.unsqueeze(0)], 0)
            IP_score,ip_each = self.IP(dct.clone(), (predic.clone()).transpose(2,3), num_person, bat_itw, self.AP) # input ori can be the predic from last , so we get 
            for i in range(num_person):
                people_feature = people_feature_all[:, i, :, :]#dct[:, i, :, :]
                filter_feature = people_feature.clone().detach()
                Pe = self.PE.forward(idx=i)
                feature_att = self.Att(filter_feature, IP_score, memo2=CP_ema, embedding=Pe)
                feature_att += people_feature.clone()
                if i==0:
                    feature_att_all  =  feature_att.unsqueeze(1) 
                else:
                    feature_att_all  =  torch.cat([feature_att_all,feature_att.unsqueeze(1)],1)   
                GCN_decoder = self.GCNsQ2[k]
                feature = GCN_decoder(feature_att)
                if i == 0:
                    # level_Features = feature_att.unsqueeze(1).clone()       
                    dct_out = feature.unsqueeze(1).clone()
                    feature = torch.matmul(self.idct, feature)
                    feature = feature.transpose(1, 2)
                    predic = feature.unsqueeze(1).clone()
                else:
                    # level_Features = torch.cat([level_Features, feature_att.unsqueeze(1).clone()], 1)
                    dct_out = torch.cat([dct_out, feature.unsqueeze(1).clone()], 1)
                    feature = torch.matmul(self.idct, feature)
                    feature = feature.transpose(1, 2)
                    predic = torch.cat([predic, feature.unsqueeze(1).clone()], 1)
            dct = dct_out
            # results_feat.append(level_Features)
            people_feat_results.append(dct)        
            people_ip_results.append(ip_each)
            people_att.append(feature_att_all)
            results.append(predic.clone())   

        # concatenated = torch.cat(results, dim=3)  # shape [B, T, D * N]
        # predic = self.linear(concatenated)
        if mode == "adaptive":#and training:
            B, P, T, J= results[0].shape
            device = results[0].device
            M = self.k_levels

            # if epoch > 30:
            filtered_results = [r[:,:,self.opt.frame_in:,:] for r in results]
            # filtered_people_feat_results= [r[:,:,:,self.opt.frame_in:] for r in people_feat_results]
            stacked_feats = torch.stack(filtered_results, dim=-1)
            gt_expanded = gt.unsqueeze(-1)  # -> (B, P, T, J, 1)
            gt_expanded = gt_expanded.transpose(2, 3)
            l2norm_error = torch.norm((stacked_feats - gt_expanded[:, :, self.opt.frame_in:, :,:]), dim=3) # -> (B, P, T, J, M)
            attn_logits = self.attention_mlp(l2norm_error)  #(B,P,T, M)
            attn_logits = attn_logits.view(B, P,self.opt.frame_out, M)
            def gumbel_tau_schedule(epoch, total_epochs, min_tau=0.5, max_tau = 2):
                progress = epoch / total_epochs
                tau = max_tau * (1 - progress) + min_tau * progress
                return max(0.5,tau)

            tau = gumbel_tau_schedule(epoch,80)
            # hard = False
            
            # attn_weights = F.softmax(attn_logits / tau, dim=-1)
            # tau = 1 
            
            attn_weights = F.gumbel_softmax(attn_logits, tau=tau, hard=True)

            stacked = torch.stack(filtered_results, dim=0)
            # Expand attn_weights: (M, B, P, T, J)
            attn = attn_weights.permute(3,0,1,2).unsqueeze(-1) # M,B,P,T,1
            # Multiply and sum: final shape (B, P, T, J)
            fused = (attn * stacked).sum(dim=0)
            # print(fused.shape)
            predic = torch.cat([results[0][:,:,:self.opt.frame_in,:], fused], 2)

            with torch.no_grad():
                print(" Data from the Model")
                max_idx = attn_weights.argmax(dim=-1)  # shows which level was chosen
                # for i in range(3):
                #     print(f"Level {i} chosen: {(max_idx == i).float().mean().item():.2f}")
                # print("this is the mean of each level",max_idx.float().mean())
                # print("attn_weights mean/std:", attn_weights.mean().item(), attn_weights.std().item())

                # print("True selected ", attn_weights)
             #B,P,J,T,K,F  
                # with open("tensor_4d.txt", "w") as f:
                # # l2norm = torch.norm((IP_score), dim=3)
                #     print(W_A_scores.shape)
                #     tensor_np = W_A_scores[:,:,:,self.opt.frame_in:,:,:].cpu()
                #     for i in range(1):
                #         for j in range(tensor_np.shape[1]):
                #             for k in range(1):
                #                 for t in range(1):
                #                     f.write(f"Slice [{i},{j},{k},{t}]:\n")
                #                     np.savetxt(f, tensor_np[i, j, k,t])
                #                     f.write("\n")
                        #     print(bat_itw.shape)    
                    # IP_score = self.IP(people_feat_results[i].clone(), (result.clone()).transpose(2,3), num_person, bat_itw, self.AP, Viz= True)           
                    # if i == 0 :
                    #     ipscores = IP_score.unsqueeze(-1)
                    # else:
                    #     ipscores = torch.cat([ipscores,IP_score.unsqueeze(-1)],-1)    
                # with open("tensor_4d.txt", "w") as f:
                #     # l2norm = torch.norm((IP_score), dim=3)
                #     tensor_np = ipscores[:,:,:,self.opt.frame_in:,:].cpu()
                #     for i in range(1):
                #         for j in range(tensor_np.shape[1]):
                #             for k in range(1):
                #                 for m in range(tensor_np.shape[3]):
                #                     f.write(f"Slice [{i},{j},{k},{m}]:\n")
                #                     np.savetxt(f, tensor_np[i, j, k,m])
                #                     f.write("\n")
            mean_attn = attn_weights.mean(dim=(0, 1, 2))  # (M,)
            uniform = torch.full_like(mean_attn, 1.0 / mean_attn.size(-1))
            diversity_loss = F.kl_div(mean_attn.log(), uniform, reduction='batchmean')
            entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean()  # scalar
            loss = self.w_cb * k_loss + 0.01 * entropy + 0.01*diversity_loss
        if training or not FineTune:
            B, P, T, J= results[0].shape
            device = results[0].device
            M = self.k_levels

            # if epoch > 30:
            filtered_results = [r[:,:,self.opt.frame_in:,:] for r in results]
            # filtered_people_feat_results= [r[:,:,:,self.opt.frame_in:] for r in people_feat_results]
            stacked_feats = torch.stack(filtered_results, dim=-1)
            gt_expanded = gt.unsqueeze(-1)  # -> (B, P, T, J, 1)
            gt_expanded = gt_expanded.transpose(2, 3)
            l2norm_error = torch.norm((stacked_feats - gt_expanded[:, :, self.opt.frame_in:, :,:]), dim=3) # -> (B, P, T, M)
            attn_logits = (1/(1+l2norm_error))
            tau= 0.5
            attn_weights = F.gumbel_softmax(attn_logits, tau=tau, hard=True)
            stacked = torch.stack(filtered_results, dim=0)
            # Expand attn_weights: (M, B, P, T, J)
            attn = attn_weights.permute(3,0,1,2).unsqueeze(-1) # M,B,P,T,1
            # Multiply and sum: final shape (B, P, T, J)
            fused = (attn * stacked).sum(dim=0)
            # print(fused.shape)
            predic = torch.cat([results[0][:,:,:self.opt.frame_in,:], fused], 2)

            with torch.no_grad():
                print(" Data from the Model")
                max_idx = attn_weights.argmax(dim=-1)  # shows which level was chosen

            # mean_attn = attn_weights.mean(dim=(0, 1, 2))  # (M,)
            # uniform = torch.full_like(mean_attn, 1.0 / mean_attn.size(-1))
            # diversity_loss = F.kl_div(mean_attn.log(), uniform, reduction='batchmean')
            # entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean()  # scalar
            loss = 0
        with torch.no_grad():
                ipscores = []
                Weight_scores_acrossLevels = []
                AMP_scores_acrossLevels = []

                for i,result in enumerate(results):
                    for bat_idx in range(self.opt.batch_size):
                        itw = LP.Trajectory_Weight(self.opt, result.transpose(2,3)[bat_idx],Viz = True)
                        if bat_idx == 0:
                            Weight_scores = itw.unsqueeze(0)
                        else:
                            Weight_scores = torch.cat([Weight_scores, itw.unsqueeze(0)], 0)

                    AMP_scores,_ =self.AP(people_feat_results[i].clone(),Viz = True)    
                    if i == 0:
                        Weight_scores_acrossLevels = Weight_scores.unsqueeze(-1)
                        AMP_scores_acrossLevels = AMP_scores.unsqueeze(-1)
                    else:
                        Weight_scores_acrossLevels = torch.cat([Weight_scores_acrossLevels,Weight_scores.unsqueeze(-1)],-1)
                        AMP_scores_acrossLevels = torch.cat([AMP_scores_acrossLevels,AMP_scores.unsqueeze(-1)],-1)
                W_A_scores = torch.cat([Weight_scores_acrossLevels.unsqueeze(2).repeat(1,1,self.num_kpt,1,1).unsqueeze(-1),AMP_scores_acrossLevels.unsqueeze(-1)],-1)
                



        if self.dataset == "Mocap" or self.dataset == "CHI3D" or self.dataset == "Wusi":
            if training:#W_A_scores.permute(0,1,3,2,4,5)[:,:,self.opt.frame_in:,:,:,:]
                # return predic,results,loss ,attn_weights , people_ip_results ,people_att ,l2norm_error
                return predic,results,loss,attn_logits ,attn_weights , people_ip_results ,people_att ,l2norm_error
            elif not training or not FineTune:    
                return predic,results, attn_weights,W_A_scores.permute(0,1,3,2,4,5)[:,:,self.opt.frame_in:,:,:,:], max_idx.float().mean() ,people_ip_results ,people_att
        elif self.dataset == "Human3.6M":
            return predic[:, 0, :, :], loss

    def mix_loss(self, predic, gt, other_loss):

        gt = gt.transpose(2, 3)
        bs, np, sql, _ = gt.shape
        spacial_loss_pred = 0
        spacial_loss_ori = 0
        spacial_loss_pred = spacial_loss_pred+torch.mean(torch.norm((predic[:, :, self.opt.frame_in:, :] - gt[:, :, self.opt.frame_in:, :]), dim=3))
        spacial_loss_ori = spacial_loss_ori + torch.mean(torch.norm((predic[:, :, :self.opt.frame_in, :] - gt[:, :, :self.opt.frame_in, :]), dim=3))
        spacial_loss = spacial_loss_pred + spacial_loss_ori * 0.1

        temporal_loss = 0

        # for r in results:
        for idx_person in range(np):

            
            tempo_pre = self.timecon(predic[:, idx_person, :, :].unsqueeze(1))
            tempo_ref = self.timecon(gt[:, idx_person, :, :].unsqueeze(1))
            
            temporal_loss += torch.mean(torch.norm(tempo_pre-tempo_ref, dim=3))

        loss = self.w_sp * spacial_loss + self.w_tp * temporal_loss + other_loss

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
class FusionModule(nn.Module):
    def __init__(self,k_levels, opt, hidden_layer=256, keypoints=45,n_head=4, num_layers=2):
        super().__init__()
        self.the_extractor = feats_extractor.FeatsExtractor(opt)
        self.err_modeling = False
        self.multi_level_err_modeling = True
        self.k_levels = k_levels +1
        self.opt = opt
       
        if self.err_modeling:
            self.error_modeling_1= nn.Sequential(
                nn.Linear(4, hidden_layer),
                nn.ReLU(),   
            )
            self.error_modeling_feats= nn.Sequential(
                nn.Linear(4, hidden_layer),
                nn.ReLU(),   
                nn.Linear(hidden_layer, hidden_layer//2)
            )
            self.error_modeling_2= nn.Sequential(
                nn.Linear(self.k_levels*hidden_layer, hidden_layer),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_layer, hidden_layer//2),
                
            )
            self.error_modeling_3= nn.Sequential(
                nn.Linear(self.opt.frame_out, hidden_layer//2),          
            )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_layer//2, 
                nhead=n_head, 
                batch_first=True  # (B, T, hidden_dim)
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.out_proj =nn.Linear(hidden_layer//2, self.k_levels*3)
            self.out_proj_2 =nn.Linear(hidden_layer//2, self.opt.frame_out) 
 
            self.attention_mlp = nn.Sequential(
            nn.Linear(self.k_levels*3, 128),
            nn.LayerNorm(128),             # Normalizes across features
            nn.ReLU(),
            nn.Dropout(0.3),            # Helps avoid overconfidence

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, self.k_levels)         # Final logits over K levels
        )
        if self.multi_level_err_modeling:
            
            self.error_modeling_feats=nn.ModuleList([ nn.Sequential(
            nn.Linear(17*self.k_levels, 64),
            nn.LayerNorm(64),             # Normalizes across features
            nn.ReLU(),
            nn.Dropout(0.3),            # Helps avoid overconfidence

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 3) 
                # nn.Linear(4, hidden_layer),
                # # nn.ReLU(),   
                # # nn.Linear(hidden_layer, hidden_layer//2)
            )for _ in range(1)])

            self.result_score_time=nn.ModuleList([ nn.Sequential(
            nn.Linear(17*self.k_levels, 32),
            nn.LayerNorm(32),             # Normalizes across features
            nn.ReLU(),
            nn.Dropout(0.3),            # Helps avoid overconfidence

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(16, 3) 
                # nn.Linear(4, hidden_layer),
                # # nn.ReLU(),   
                # # nn.Linear(hidden_layer, hidden_layer//2)
            )for _ in range(self.opt.frame_out//5)])

            self.upscaler = nn.Linear(42,128)
            # self.error_modeling_time_feats = nn.Linear(self.opt.frame_out, hidden_layer*2)
           
            self.encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(
            d_model=128,
            nhead=16,
            batch_first=True
            )
            for _ in range(self.opt.k_level+1)
        ])
            self.pos_emb = nn.Parameter(torch.randn(1, self.opt.frame_out, 128))
            # self.GCN_spatial = GCN.GCN(input_feature=5,
            #                  hidden_feature=128,
            #                  p_dropout=0.3,
            #                  num_stage=2,
            #                  node_n=45)#2
            self.out_proj = nn.ModuleList([nn.Linear(hidden_layer,3) for _ in range(self.opt.k_level+1)])
            # self.out_proj_2 =nn.Linear(hidden_layer*2, self.opt.frame_out) 
            self.norm_att = nn.LayerNorm(self.opt.frame_out)
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

        else:
            self.in_proj = nn.Linear(self.k_levels * 5, hidden_layer)  # (M × 4) → hidden_dim
            self.joint_mlp = nn.Sequential(
                nn.Linear(hidden_layer * keypoints, hidden_layer),
                nn.ReLU(),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_layer, 
                nhead=n_head, 
                batch_first=True  # (B, T, hidden_dim)
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.out_proj = nn.Linear(hidden_layer, self.k_levels)
        self.timecon = TimeCon.timecon_plus()

        self.alpha = 0.5 
    def forward(self, results, gt ,dct, epoch,predic_gt = None, teacher_logits = None, teacher_weights = None,Feats = None, Train= True, ip = None , Att_feats = None, l2_error = None):
        
        # error_modeling = True
       
        
        if Train :
            training = True
        else:
            training = False
        # print("this is d7k ", results.shape)
        B, P, T, J = results[0].shape
        M = self.k_levels

        # new_feats = []
        # new_feats =self.the_extractor.get_feats(results.clone())
        # for m in range(M):
        #     new_feats.append()
        # new_feats=torch.stack(new_feats, dim=-2) 

        # print(new_feats.shape)
        ##habalela   
        # detached_list = [t.detach() for t in results]
        # filtered_results = [r[:,:,self.opt.frame_in-2:,:] for r in detached_list]
        # dct_filtered = [torch.matmul(dct, r[:,:,self.opt.frame_in:,:].permute(0,1,3,2)) for r in detached_list]

        # stacked_feats = torch.stack(filtered_results, dim=-1) # B,P,T,J,K    
        # dct_filtered = torch.stack(dct_filtered, dim=-1)
        # displacements = (stacked_feats[:, :, 1:, :, :] - stacked_feats[:, :, 0:1, :, :])[:, :, 1:, :, :].mean(3) # (B, P, T-1, J, M)
        # stacked_feats = stacked_feats[:, :, 2:, :, :]
        # dct_filtered = dct_filtered.permute(0,1,3,2,4).mean(3)
        from itertools import combinations

        pairs = list(combinations(range(M), 2))
        diffs = []
        diffs_2 = []
        for m in range(M):
            pair = pairs[m]
            print(pair[0],pair[1],torch.norm(results[pair[0]]-results[pair[1]],dim=-1).shape)
            diff = torch.norm(results[pair[0]]-results[pair[1]],dim=-1)
            diffs.append(diff)
        diffs = torch.stack(diffs, dim = -1)
        btngan = torch.zeros(diffs.shape, device =diff.device )
        for m in range(M):
            x,y = pairs[m]
            btngan[...,x] += diffs[...,m]  
            btngan[...,y] += diffs[...,m]  
        print("sum of levels differences between each others",btngan.shape) 
        btngan = btngan[:,:,self.opt.frame_out:]
        # print(diffs[0,0,26])
        # print(btngan[0,0,1])


        FFT = []
        for m in range(M):
            B, P, T, J = results[m].shape
            pose = results[m,:,:,self.opt.frame_out:].view(B,P,self.opt.frame_out,J//3,3) #bptjc
            # Transpose to (B, P, J, C, T) to FFT across time
            motion = pose.permute(0, 1, 3, 4, 2)

            # Apply real FFT (over time dimension)
            fft = torch.fft.rfft(motion, dim=-1)  # shape (B, P, J, 3, F)
            magnitude = fft.abs()  # Get magnitude of frequency components

            # Average across 3D channels (x, y, z) → shape: (B, P, J, F)
            freq_features = magnitude.mean(dim=3).mean(dim=2).unsqueeze(2).repeat(1,1,self.opt.frame_out,1)
            FFT.append(freq_features)
        FFT = torch.stack(FFT,dim = 0).permute(1,2,3,0,4)


        hi =diffs.argmin(dim = -1)
        kofta = []
        #kofta_2 = []
        for m in range(M):
            x = torch.abs(results[m,:,:,:,:].mean(-1)-results[:,:,:,:,:].mean(-1).mean(0))
            #y = torch.abs(ip[m,:,:,:,:].mean(-2)-ip[:,:,:,:,:].mean(-2).mean(0))
            kofta.append(x)
           # kofta_2.append(y)
        kofta = torch.stack(kofta, dim = -1)[:,:,self.opt.frame_out:]
       # kofta_2 = torch.stack(kofta_2, dim = -1)[:,:,self.opt.frame_out:]
        #print(kofta_2.shape)
        # print(kofta.shape)  
        #ip = ip.permute(1,2,4,3,0)[:,:,self.opt.frame_in:,:,:].mean(3) 
        #Att_feats = Att_feats.permute(1,2,4,3,0)[:,:,self.opt.frame_in:,:,:].mean(3)
        # print(kofta[5,0,:].argmin(dim = -1))
        # print(kofta_2[5,0,:].argmin(dim = -1))
        # print(ip[5,0,:].argmin(dim = -1))
        #print()
        # h = (kofta[0,0,:]*kofta_2[1,0,:]).argmin(dim = -1)
        # # b = 1
        # # p = 0
        # sum = 0
        # t1 = 20
        # t2 = 25
        # results = results.permute(1,2,3,4,0).mean(-2)[:,:,self.opt.frame_out:]
        # print(results.shape)
        # print((results[:,:,:,:].mean(-1)).shape)
        # for b in range(60,62,1):
        #     for p in range(1):
        #         # for t in range(self.opt.frame_out):
        #         print(displacements.shape)
        #         print(torch.cat([btngan[b,p,t1:t2]*kofta[b,p,t1:t2],(results[:,:,:,:].mean(-1))[b,p,t1:t2].unsqueeze(-1),displacements[b,p,t1:t2],dct_filtered[b,p,t1:t2]],dim = -1))
        #         print(l2_error[b,p,t1:t2])
                
        #         h3 = ((btngan[b,p,t1:t2]*(kofta[b,p,t1:t2])/torch.abs(displacements[b,p,t1:t2]))).argmin(dim = -1)
        #         # h4 = (btngan[b,p,:t]*(results[b,p,:t]*(kofta[b,p,:t]))).argmin(dim = -1)
        #         # h_2 = (results[b,p,:t]*(kofta[b,p,:t])).argmin(dim = -1)
        #         h_1= l2_error[b,p,t1:t2].argmin(dim = -1)
        #         print(h3, h_1)
        #         # quit()
        #         sum +=(h_1==h3).sum()
        # print(sum)        
        # # i = 38
        # # print(torch.abs(results[0,0,1,i].mean(-1)-results[:,0,1,i].mean(-1).mean(0)))
        # # print(torch.abs(results[1,0,1,i].mean(-1)-results[:,0,1,i].mean(-1).mean(0)))
        # # print(torch.abs(results[2,0,1,i].mean(-1)-results[:,0,1,i].mean(-1).mean(0)))

        # from collections import Counter
        # print("what ios this ",Counter(hi.view(-1).tolist()))
        # quit()
        ##habalela

        detached_list = [t.detach() for t in results]
        filtered_results = [r[:,:,self.opt.frame_in-2:,:] for r in detached_list]
        dct_filtered = [torch.matmul(dct, r[:,:,self.opt.frame_in:,:].permute(0,1,3,2)) for r in detached_list]

        stacked_feats = torch.stack(filtered_results, dim=-1) # B,P,T,J,K    
        dct_filtered = torch.stack(dct_filtered, dim=-1)
        displacements = (stacked_feats[:, :, 1:, :, :] - stacked_feats[:, :, 0:1, :, :])[:, :, 1:, :, :].mean(3) # (B, P, T-1, J, M)
        stacked_feats = stacked_feats[:, :, 2:, :, :]
        dct_filtered = dct_filtered.permute(0,1,3,2,4).mean(3)

        Att_feats = Att_feats.permute(1,2,4,3,0)[:,:,self.opt.frame_in:,:,:]
        ip = ip.permute(1,2,4,3,0)[:,:,self.opt.frame_in:,:,:]#.mean(3)
        # print(stacked_feats.shape,dct_filtered.shape,Att_feats.shape,displacements.shape,ip.shape)
        motion_features = torch.stack([dct_filtered,displacements,btngan,kofta],-1)#.mean(3) #Att_feats displacements stacked_feats
        # print(motion_features.shape)
        motion_features = torch.cat([motion_features,FFT],dim = -1)
        # print("hi",motion_features.shape)
        
         #(B, P, 25, J, M, 4

        if training:
            with torch.no_grad():
                # err = ((outputs_stack - gt.unsqueeze(3)) ** 2).mean(dim=-1)  # (B, T, J, N)
                gt_expanded = gt.unsqueeze(-1)  # -> (B, P, T, J, 1)
                gt_expanded = gt_expanded.transpose(2, 3)
                print(stacked_feats.shape,gt_expanded.shape,dct.shape,stacked_feats.shape, dct_filtered.shape)
                l2norm_error = torch.norm((stacked_feats - gt_expanded[:, :, self.opt.frame_in:, :,:]), dim=3) #B,P,T,K
                # Oracle 1-hot mask: 1 where level has lowest error
                best_level = torch.argmin(l2norm_error, dim=-1)  # (B, P, T)
                print("shaep pf the best",best_level.shape)
                for i in range(3):
                    print(f"Level {i} chosen best: {(best_level == i).float().mean().item():.2f}")
                gt_selector_mask = F.one_hot(best_level, num_classes=self.k_levels).float()
                print("shaep pf the mask",gt_selector_mask.shape)
        # stacked_feats = stacked_feats.mean(0) # P,T,J,K

        # motion_features = torch.norm((motion_features), dim=3)
        

        # motion_features = motion_features#.mean(dim=3)

        if self.err_modeling:
            motion_features =  self.error_modeling_1(motion_features)
            motion_features = motion_features.reshape(B, P, self.opt.frame_out, -1)
            motion_features = self.error_modeling_2(motion_features)
            motion_features = motion_features.reshape(B ,P, M,3,self.opt.frame_out) 
            motion_features = self.error_modeling_3(motion_features)
            motion_features = motion_features.reshape(B*P, -1,64) 
            motion_features = self.transformer(motion_features)
           
            motion_features = self.out_proj_2(motion_features).view(B, P, self.opt.frame_out, -1)
            attn_scores = self.out_proj(motion_features).view(B, P, self.opt.frame_out, M*3)  
            if training:
                sorted_errors, indices = torch.sort(l2_error, dim=-1)  # sort over M
                labels = torch.full_like(l2_error, fill_value=M, dtype=torch.long)
                for i in range(M):
                    if i >0:
                        level_indices = indices[..., i].unsqueeze(-1)
                        labels.scatter_(-1, level_indices, 1)
                    else:    
                        level_indices = indices[..., i].unsqueeze(-1)
                        labels.scatter_(-1, level_indices, i)
                loss_fn = nn.CrossEntropyLoss()
                ce_loss = loss_fn(attn_scores.view(-1, 3), labels.view(-1))
                pt = torch.exp(-ce_loss)  # pt is the prob assigned to the true class
                err_loss = 1 * (1 - pt) ** 2 * ce_loss


            #     # err_loss = torch.mean(torch.norm((motion_features_logits- l2_error)))
            
            # # motion_features_logits # B,P,T,M
            # probs = torch.sigmoid(attn_scores)
            # low_probs = probs[..., 0]
            # best_low_idx = low_probs.argmax(dim=-1)
            # mask = torch.zeros(B, P,  self.opt.frame_out, M, device=low_probs.device)

            # # Use scatter to assign 1 to the best index
            # attn_weights=mask.scatter_(-1, best_low_idx.unsqueeze(-1), 1.0)
            # attn = attn_weights.permute(3,0,1,2).unsqueeze(-1)
            attn_scores = self.attention_mlp(attn_scores.detach())  #(B,P,T, M)
            attn_scores = attn_scores.view(B, P,self.opt.frame_out, M)
            def gumbel_tau_schedule(epoch, total_epochs, min_tau=0.5, max_tau = 4):
                    progress = epoch / total_epochs
                    tau = max_tau * (1 - progress) + min_tau * progress
                    return max(0.5,tau)

            tau = gumbel_tau_schedule(epoch,80)
            hard = False
                
            # attn_weights = F.softmax(attn_scores / tau, dim=-1)
            # hard = False
            
            # attn_weights = F.softmax(attn_scores / tau, dim=-1)
            # tau = 1 
            attn_weights = F.gumbel_softmax(attn_scores, tau=tau, hard=True)
            
            attn = attn_weights.permute(3,0,1,2).unsqueeze(-1) # K,B,P,T,1
           

        if self.multi_level_err_modeling:
            attn_scores = []
            
            for t in range(self.opt.frame_out//5):
                motion_level_feats = motion_features[:,:,t*5:(t+1)*5,:,:]
                motion_level_feats = motion_level_feats.reshape(-1,M*motion_level_feats.shape[-1])
                print(motion_level_feats.shape)
                # quit()
                motion_level_feats=self.result_score_time[t](motion_level_feats)
                print(motion_level_feats.shape)
                motion_level_feats = motion_level_feats.reshape(B,P,5,-1)
                if t ==0:    
                    attn_scores = motion_level_feats
                else:
                    attn_scores =torch.cat([attn_scores,motion_level_feats],2) 
            print(attn_scores.shape)




            # for m in range(1):
            #     motion_level_feats = motion_features[:,:,:,:,:]
            #     #habalelo
            #     # motion_level_feats=self.upscaler(motion_level_feats)
            #     # # motion_level_feats = motion_level_feats.view(-1, J,motion_features.shape[-1])
            #     # # motion_level_feats = self.GCN_spatial(motion_level_feats,is_output=True)
            #     # # motion_level_feats = motion_level_feats.view(B, P, self.opt.frame_out, J, -1)
            #     # motion_level_feats = motion_level_feats.reshape(B*P, self.opt.frame_out, -1)
            #     # # print("d7k",motion_level_feats.shape)

            #     # motion_level_feats = motion_level_feats + self.pos_emb
            #     # motion_level_feats= self.encoders[m](motion_level_feats).reshape(B,P, self.opt.frame_out,-1)
            #     # # print("d7k",motion_level_feats.shape)

            #     # # motion_level_feats = motion_level_feats.reshape(B ,P,-1,J,self.opt.frame_out)

            #     #  # B,P,T,hidden =64
                
            #     #  # B*P,hidden =64, hidden = 256
                    
            #     # # motion_level_feats = motion_level_feats.mean(3)
            #     # # motion_level_feats = motion_level_feats.reshape(B ,P,self.opt.frame_out,-1)
            #     #habalelo
                
            #     attn_scores =  self.error_modeling_feats[0](motion_level_feats.view(-1,M*motion_level_feats.shape[-1]))
            #     attn_scores = attn_scores.reshape(B,P,self.opt.frame_out,-1)

                # motion_level_feats = self.out_proj[m](motion_level_feats) #B,P,T,3
                # if m ==0:
                #     attn_scores = motion_level_feats.unsqueeze(3)
                # else:
                #   attn_scores =torch.cat([attn_scores,motion_level_feats.unsqueeze(3)],3) 
            # print(attn_scores.shape)    
            # quit()
  
            # attn_scores = attn_scores.mean(3)         
            
            # print("thji is the best hsape ever ", attn_scores.shape)
            # attention
            attn_scores= attn_scores.view(B, P, self.opt.frame_out, M)
            attn_scores = self.attention_mlp(attn_scores.detach())  #(B,P,T, M)
            attn_scores = attn_scores.view(B, P,self.opt.frame_out, M)
            def gumbel_tau_schedule(epoch, total_epochs, min_tau=0.5, max_tau = 4):
                    progress = epoch / total_epochs
                    tau = max_tau * (1 - progress) + min_tau * progress
                    return max(0.5,tau)

            tau = gumbel_tau_schedule(epoch,80)
            hard = False
                
            # attn_weights = F.softmax(attn_scores / tau, dim=-1)
            # hard = False
            
            # attn_weights = F.softmax(attn_scores / tau, dim=-1)
            # tau = 1 
            attn_weights = F.gumbel_softmax(attn_scores, tau=tau, hard=True)
            
            attn = attn_weights.permute(3,0,1,2).unsqueeze(-1) # K,B,P,T,1   

        else:
            motion_features = motion_features.reshape(B, P, self.opt.frame_out, M * 5)
            # print(motion_features.shape)
            motion_features = self.in_proj(motion_features)
            # motion_features = motion_features.reshape(B, P, self.opt.frame_out, -1)  # (B, P, T, hidden_dim * J)
            # motion_features = self.joint_mlp(motion_features)
            motion_features = motion_features.reshape(B * P, self.opt.frame_out, -1)            # (B*P, T, hidden_dim)
            motion_features = self.transformer(motion_features)
            attn_scores = self.out_proj(motion_features)             # (B*P, T, M)
            attn_scores = attn_scores.view(B, P, self.opt.frame_out, M)
            # def gumbel_tau_schedule(epoch, self.opt.epochs, min_tau=0.5, max_tau=5):
            #     progress = epoch / total_epochs
            #     tau = max_tau * (1 - progress) + min_tau * progress
            #     return max(0.5,tau)

            def gumbel_tau_schedule(epoch, total_epochs, min_tau=0.5, max_tau = 4):
                    progress = epoch / total_epochs
                    tau = max_tau * (1 - progress) + min_tau * progress
                    return max(0.5,tau)

            tau = gumbel_tau_schedule(epoch,80)
            hard = False
                
            # attn_weights = F.softmax(attn_scores / tau, dim=-1)
            # hard = False
            
            # attn_weights = F.softmax(attn_scores / tau, dim=-1)
            # tau = 1 
            attn_weights = F.gumbel_softmax(attn_scores, tau=tau, hard=True)
            
            attn = attn_weights.permute(3,0,1,2).unsqueeze(-1) # K,B,P,T,1
        # Multiply and sum: final shape (B, P, T, J)
        stacked = torch.stack(filtered_results, dim=0) [:,:,:,2:,:]# K,B,P,T,J
        fused = (attn * stacked).sum(dim=0)
        predic = torch.cat([detached_list[0][:,:,:self.opt.frame_in,:], fused], 2)
        with torch.no_grad():
            # probs = F.softmax(attn_logits, dim=-1)
            max_idx = attn_weights.argmax(dim=-1)  # shows which level was chosen
            # print("Selected levels weights for 1 batch (frame, joint):", attn_weights[0])  # shape: (T, J)
            for i in range(3):
                print(f"Level {i} chosen: {(max_idx == i).float().mean().item():.2f}")
            # print("this is the mean of each level",max_idx.float().mean())
            # print("attn_weights mean/std:", attn_weights.mean().item(), attn_weights.std().item())

              # (B, P, T)
            # gt_selector_mask = F.one_hot(best_level, num_classes=self.k_levels).float()
        
        if training :
            sorted_errors, indices = torch.sort(l2_error, dim=-1)  # sort over M
            # print(M)
            # return
            labels = torch.full(l2_error.shape, fill_value=self.opt.k_level, dtype=torch.long, device=l2_error.device)
            print(indices.shape,labels.shape)
            for i in range(self.opt.k_level):
                if i >0:
                    # level_indices = indices[..., i].unsqueeze(-1)
                    labels.scatter_(-1, indices[..., 1].unsqueeze(-1), 1)

                else:    
                    # level_indices = indices[..., i].unsqueeze(-1)
                    # labels.scatter_(-1, level_indices, i)
                    labels.scatter_(-1, indices[..., 0].unsqueeze(-1), 0)

            

            if   self.err_modeling or self.multi_level_err_modeling:
                # selector_loss = F.kl_div(attn_weights.log(), attn_weights_gt, reduction='batchmean')
                # distill_loss = self.kl_div(student_soft, teacher_soft) * (T * T)
                # spacial_loss_pred = torch.mean(torch.norm((predic[:, :, self.opt.frame_in:, :] - predic_gt[:, :, self.opt.frame_in:, :]), dim=3))
                T = 1
                best_level = torch.argmax(teacher_weights, dim=-1)
                student_log_probs = F.log_softmax(attn_scores / T, dim=-1)
                teacher_probs = F.softmax(teacher_logits / T, dim=-1)
                print(teacher_probs.shape,student_log_probs.shape)

                kl_loss=F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
                # ce_loss  = F.cross_entropy(
                # attn_scores.view(-1, M),  # (B×P×T, K)
                # best_level.view(-1))   
                #       # (B×P×T,)
                # selector_loss = (self.alpha*ce_loss) + ((1-self.alpha)*kl_loss)
                gt = gt.transpose(2, 3)
                pred_loss = torch.mean(torch.norm((predic[:, :, self.opt.frame_in:, :]- gt[:, :, self.opt.frame_in:, :]), dim=3))
                weight_loss =  F.softmax(torch.norm((predic[:, :, self.opt.frame_in:, :]- gt[:, :, self.opt.frame_in:, :]), dim=3),dim =-1)

                # print(weight_loss.shape,(attn_scores - ((1/(1+l2_error))) ** 2).shape)
                # quit()



                print(labels.shape,attn_scores.shape) 
                loss_fn = nn.MSELoss()
                # attn_scores = attn_scores.reshape(B,P,attn_scores.shape[2],-1)
                print(attn_scores.shape,l2_error.shape)
                # loss_1 = ((F.softmax(attn_scores,dim = -1)[:,:,20:] - F.softmax((1/(1+l2_error)),dim = -1)[:,:,20:]) ** 2).mean()
                # loss_2 = (5 * (F.softmax(attn_scores,dim = -1)[:,:,10:20] - F.softmax((1/(1+l2_error)),dim = -1)[:,:,10:20]) ** 2).mean()
                # loss_3 = (1 * (F.softmax(attn_scores,dim = -1)[:,:,0:10] - F.softmax((1/(1+l2_error)),dim = -1)[:,:,0:10]) ** 2).mean()
                # ce_loss =torch.mean(weight_loss * (attn_scores - ((1/(1+teacher_logits))) ** 2).mean(-1)) #+ loss_2 
                ce_loss = loss_fn(attn_scores,teacher_logits)
                # loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
                # # print(attn_scores )      
                # # import time

                # # # Wait for 100000 seconds
                # # time.sleep(2)
                        
                
                
                # ce_loss = loss_fn(attn_scores.view(-1, 3), labels.view(-1))
                prob =attn_scores# F.softmax(attn_scores, dim=-1)
                # print(prob) 
                # probs = F.softmax(attn_scores, dim=-1)  # (B, P, T, M, 3)

                # Compute entropy
                # entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)  # shape: (B, P, T, M)

                # Minimize entropy → encourage confidence
                # entropy_loss = entropy.mean() 
                print(prob.shape , (1/(1+l2_error)).shape)
                prob = prob.reshape(-1,3)
                preds = prob.argmax(dim=1)  # shape: (B*...)
                d7k = (teacher_logits).reshape(-1, 3).argmax(dim =1)

                print(preds.shape, d7k.shape)
                # Get true labels
                targets = labels.view(-1)  # shape: (B*...)
                
                # Calculate accuracy
                correct = (preds == d7k).sum().item()
                print(correct)
                total = d7k.numel()
                accuracy = correct / total

                print(f"Accuracy: {accuracy:.4f}")
                # quit()

                from collections import Counter
                print("what ios this ",Counter(labels.view(-1).tolist()))
                #pt = torch.exp(-ce_loss)  # pt is the prob assigned to the true class
                err_loss = ce_loss #+ 0.1 * entropy_loss#1 * (1 - pt) ** 2 * ce_loss


                temporal_loss = 0
                np = predic.shape[1]
                # for r in results:
                for idx_person in range(np):

                    
                    tempo_pre = self.timecon(predic[:, idx_person, :, :].unsqueeze(1))
                    tempo_ref = self.timecon(gt[:, idx_person, :, :].unsqueeze(1))
                    
                    temporal_loss += torch.mean(torch.norm(tempo_pre-tempo_ref, dim=3))
                print(err_loss)

                return predic,(pred_loss + temporal_loss)+ err_loss#selector_loss+ (0.5*kl_loss)  +
            else:
                return predic, err_loss    
        else:
           
            return predic, max_idx.float().mean()


class Discriminator(nn.Module):
    def __init__(
            self, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=128, d_model=128, d_inner=1024,
            n_layers=3, n_head=8, d_k=64, d_v=64, dropout=0.2, n_position=100, d_hidden=1024):

        super().__init__()
        self.d_model=d_model
        self.encoder = Encoder(
                n_position=n_position,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                pad_idx=src_pad_idx, dropout=dropout)
        self.fc1 = nn.Linear(45, d_hidden)
        self.bn1 = nn.BatchNorm1d(d_hidden)
        self.fc2 = nn.Linear(d_hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        #score learning

        self.encoder_score = Encoder(
                n_position=n_position,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                pad_idx=src_pad_idx, dropout=dropout)
        self.fc1_score = nn.Linear(45, d_hidden)
        self.bn1_score = nn.BatchNorm1d(d_hidden)
        self.fc2_score = nn.Linear(d_hidden, 1)
    
    def forward(self, x_in, n_person=5):
        x, *_ = self.encoder(x_in, n_person=n_person, src_mask=None, global_feature=True)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def forward_score(self, x_in, n_person=5):
        BP,T,J = x_in.shape
        x, *_ = self.encoder(x_in, n_person=n_person, src_mask=None, global_feature=True)
        x = self.fc1_score(x.detach())
        # print(x.shape)
        x = x.view(-1,x.shape[-1],x.shape[-2])
        x = self.bn1_score(x)
        x = x.view(-1,x.shape[-1],x.shape[-2])
        x = self.fc2_score(x)
        x = x.sigmoid()
        return x    

    def calc_gen_loss(self, x_gen):
        # print("this is the disrm logits 1",x_gen.shape)
        fake_logits = self.forward(x_gen)
        loss = discriminator_loss.disc_l2_loss(fake_logits)
        return loss
    
    def get_scores(self, x_gen):
        fake_logits = self.forward(x_gen)
        fake_scores = self.forward_score(x_gen)
        return fake_logits.view(-1, 5), fake_scores.view(-1, 5,25)

    def calc_dis_loss(self, x_gen, x_real, att_logits = None):
        fake_logits = self.forward(x_gen)
        real_logits = self.forward(x_real)
        fake_scores = self.forward_score(x_gen)
        loss_score = nn.MSELoss()
        output = loss_score(fake_scores[:,25:,:], att_logits) # error score add it later , dont forget also to add att_logits
        loss = discriminator_loss.adv_disc_l2_loss(fake_logits, real_logits)
        return loss + output        
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn      
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q = q + residual

        q = self.layer_norm(q)

        return q, attn


def disc_l2_loss(disc_value):
    
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def adv_disc_l2_loss(fake_disc_value, real_disc_value):
    kb = fake_disc_value.shape[0]
    ka = real_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la + lb

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x = x + residual

        x = self.layer_norm(x)

        return x

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        self.register_buffer('pos_table2', self._get_sinusoid_encoding_table(n_position, d_hid))
        
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self,x,n_person):
        p=self.pos_table[:,:x.size(1)].clone().detach()
        return x + p

    def forward2(self, x, n_person):
        p=self.pos_table2[:, :int(x.shape[1]/n_person)].clone().detach()
        p=p.repeat(1,n_person,1)
        return x + p


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200):

        super().__init__()
        self.position_embeddings = nn.Embedding(n_position, d_model)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        
    def forward(self, src_seq,n_person, src_mask, return_attns=False, global_feature=False):
        enc_slf_attn_list = []
        if global_feature:
            enc_output = self.dropout(self.position_enc.forward2(src_seq,n_person))
        else:
            enc_output = self.dropout(self.position_enc(src_seq,n_person))
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,