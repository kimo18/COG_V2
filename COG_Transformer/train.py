import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
sys.path.append('..')
# print(sys.path)
# from tensorboardX import SummaryWriter
from utils import other_utils as util
# from utils import View_skeleton as view3d
# from IPython import embed
from tqdm import tqdm
import matplotlib.pyplot as plt
from option.option_Mocap import Options
# from Utils import util, data_utils, vis_2p
# from Utils.rigid_align import rigid_align_torch
from utils.dataset import MPMotion
from model import IAFormer as model

# from torchstat import stat
from collections import OrderedDict
# import torchvision.models as models
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# from model import xiao_model_codebook
import random
import os
import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
Viz_scores = False
Only_adaptive = False
CSV = False
def seed_init(seed,device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)   
    torch.cuda.manual_seed(seed) 

    torch.cuda.set_device(device=device)
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)
    print("The using GPU is device {0}".format(device))
    # torch.autograd.set_detect_anomaly(True)


def main(opt,wandb_args):
    device = opt.cudaid
    seed_init(1234567890, device)
    if opt.mode == 'train':
        print('>>> DATA loading >>>')
        dataset = MPMotion(opt.train_data, mode = "Train")
        eval_dataset = MPMotion(opt.test_data, mode = "Test")
    
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        eval_data_loader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        #Wandb Init


    elif opt.mode == 'test':
        eval_dataset = MPMotion(opt.test_data, mode = "Test")
        data_loader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    in_features = opt.in_features 
    nb_kpts = int(in_features)  # number of keypoints

    print('>>> MODEL >>>')
    if opt.model == 'IAFormer':
        net_pred = model.IAFormer(seq_len=opt.seq_len, d_model=opt.d_model, opt=opt, num_kpt=nb_kpts, dataset="Mocap", k_levels = opt.k_level , share_d = False)
        discriminator = model.Discriminator(d_word_vec=45, d_model=45, d_inner=opt.d_inner_d, d_hidden=opt.d_hidden,
                n_layers=3, n_head=8, d_k=32, d_v=32, dropout=opt.drop_out)
        adaptive_selector = model.FusionModule(opt.k_level, opt, hidden_layer=128, keypoints=nb_kpts,n_head=4, num_layers=2)             


    net_pred.cuda()
    discriminator.cuda()
    adaptive_selector.cuda()
    # hyperparameter
    lr_now = opt.lr_now
    saved_lr = opt.lr_now

    start_epoch = 1

    print(">>> total params: {:.2f}M".format((sum(p.numel() for p in net_pred.parameters()) / 1000000.0)+(sum(p.numel() for p in discriminator.parameters()) / 1000000.0)+(sum(p.numel() for p in adaptive_selector.parameters()) / 1000000.0)))
    

    if opt.mode == "test":
        if '.pth.tar' in opt.ckpt:
            model_path_len = opt.ckpt
        elif opt.test_epoch is not None:
            model_path_len = '{}/ckpt_epo{}.pth.tar'.format(opt.ckpt, opt.test_epoch)
        else:
            model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)

        print(">>> loading ckpt from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        lr_now = ckpt['lr']

        net_pred.load_state_dict(ckpt['state_dict'])
        
        print(">>> ckpt loaded (epoch: {} | err: {} | lr: {})".format(ckpt['epoch'], ckpt['err'], lr_now))


    if opt.mode == 'train': #train
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
        optimizer_d = optim.Adam(filter(lambda x: x.requires_grad, discriminator.parameters()), lr=opt.lr_now)
        #optimizer_s = optim.Adam(filter(lambda x: x.requires_grad, adaptive_selector.parameters()), lr=opt.lr_now)
        optimizer_s = optim.AdamW(filter(lambda x: x.requires_grad, adaptive_selector.parameters()), lr=1e-4, weight_decay=1e-2)
        

        util.save_ckpt({'epoch': 0, 'lr': lr_now, 'err': 0, 'state_dict': net_pred.state_dict(), 'optimizer': optimizer.state_dict()}, 0, dataset=opt.dataset,opt=opt)
        # writer = SummaryWriter(opt.tensorboard)
        mpjpe_flag = 10000
        
        best_val_loss = float('inf')
        if not Only_adaptive:
            wandb.init(
                project= "testing groundtruth",
                config=wandb_args,
                name="COG_Dis_lvl2_Adaptive(test 96 batch)"
        )
            for epo in tqdm(range(1, 80 + 1)):
                ret_train = run_model(nb_kpts, net_pred , opt.batch_size,discrim = discriminator, optimizer = optimizer, optimizer_d = optimizer_d, data_loader=data_loader, opt=opt, epo=epo )
                mpjpe_mean, ape_mean = eval(opt, net_pred, eval_data_loader, nb_kpts, epo)

                # or -inf if tracking accuracy
            # Save only if improved
                val_loss = mpjpe_mean
                if   val_loss< best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epo,
                        'model_state_dict': discriminator.state_dict(),
                        'optimizer_state_dict': optimizer_d.state_dict(),
                        'val_loss': val_loss
                    }, 'discrim_checkpoint_onlychoose_least.pth')

                    torch.save({
                        'epoch': epo,
                        'model_state_dict': net_pred.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, 'best_model_checkpoint_onlychoose_least.pth')
                

                util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / 50))
                lr_now = util.lr_decay_mine(optimizer_d, lr_now, 0.1 ** (1 / 50))
            wandb.finish()
        wandb.init(
                project= opt.wandb_proj_name,
                config=wandb_args,
                name="COG_Dis_lvl2_Adaptive(test 96 batch)"
        )
        
        for epo in tqdm(range(1, 80 + 1)):
            # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            # Load checkpoint
            checkpoint = torch.load('best_model_checkpoint.pth')
            net_pred.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            net_pred.cuda()

            checkpoint = torch.load('discrim_checkpoint_onlychoose_least.pth')
            discriminator.load_state_dict(checkpoint['model_state_dict'])
            optimizer_d.load_state_dict(checkpoint['optimizer_state_dict'])
            discriminator.cuda()

            ret_train = run_model(nb_kpts, net_pred, opt.batch_size,selector = adaptive_selector, optimizer = optimizer, optimizer_s = optimizer_s, data_loader=data_loader, opt=opt, epo=epo , FineTune = True)
            mpjpe_mean, ape_mean = eval(opt, net_pred, eval_data_loader, nb_kpts, epo, selector = adaptive_selector, FineTune = True)
            # saved_lr = util.lr_decay_mine(optimizer_s, saved_lr, 0.1 ** (1 / 50))


    else: #test

        run_model(nb_kpts, net_pred, opt.batch_size, data_loader=data_loader, opt=opt)

def run_model(nb_kpts, net_pred, batch_size, selector = None, discrim =None, optimizer=None, optimizer_d = None, optimizer_s = None, data_loader=None, opt=None, epo=0 , FineTune = False):
    device = opt.cudaid
    n = 0
    if opt.mode == 'train': #train
        loss_train = 0
        losses_dis = []
        losses_recon = []
        losses_gail = []
        losses_all = []
        losses_sum = []
        B = batch_size
        for batch_idx, (x, y) in enumerate(data_loader): # in_n + kz

            torch.cuda.empty_cache()
            if np.shape(x)[0] < opt.batch_size:
                continue #when only one sample in this batch
            n += batch_size

            x = x.float().cuda()
            y = y.float().cuda()

            x_c = x.clone().detach()
            x_d = x.clone().detach()
            y_c = y.clone().detach()

            if not FineTune and not Only_adaptive:
                net_pred.train()
                discrim.train()
                data_out,results, other_loss, attention_logits_gt , attention_weights_gt ,people_ip_results ,people_att ,l2norm_error= net_pred(x_c, y_c,epo)

                loss_sum = torch.tensor(0).cuda()
                loss_dis_sum = torch.tensor(0).cuda()

                # Train Discriminator
                for k in range(0, opt.k_level + 1):
                    result_k = results[k]
                    real_full = x_d.permute(0,1,3,2).view(-1,opt.seq_len,45) 
                    pred_full = result_k.view(-1,opt.seq_len,45).detach()             
                        
                    for param in discrim.parameters():
                        param.requires_grad = True
                    loss_dis = discrim.calc_dis_loss(pred_full, real_full,att_logits = attention_logits_gt[:,:,:,k].view(-1,opt.frame_out,1).detach() )
                    optimizer_d.zero_grad()
                    loss_dis.backward()
                    optimizer_d.step()
                    # loss_dis_sum = loss_dis_sum + loss_dis
                    # losses_dis.append(loss_dis.item() / B)
                    fake_logits,fake_scores = discrim.get_scores(pred_full)
                    print(fake_scores[0,:,1].detach(),attention_logits_gt[0,:,1,k].detach())

                
                loss_sum = torch.tensor(0).cuda()

                # for inverse_k in range(opt.k_level , opt.k_level-1, -1):
                #     print(f"this is inversek = {inverse_k}")
                #     if inverse_k != opt.k_level:
                #         data_out,results, other_loss = net_pred(x_c, y_c,epo)
                #         weight = 0.1
                #     else:
                #         weight = 1    
                #     # if inverse_k == opt.k_level:
                #     #     data_out,results, other_loss , _= net_pred(x_c, y_c,epo,inverse_level)

                for k in range(opt.k_level, opt.k_level + 1):
                    print(k)
                    bc = (k == opt.k_level)
                    if bc:
                        loss_l2 = net_pred.mix_loss(data_out,y_c,other_loss) 
                        loss_recon = loss_l2
                        losses_recon.append(loss_recon.item()/B)
                    else:
                        loss_recon = torch.tensor(0).cuda()      

                    result_k = results[k]                                                                          
                    pred_full_grad = result_k.view(-1,opt.seq_len,45)                     
                    

                    for p in discrim.parameters():
                        p.requires_grad = False
                    loss_gail = discrim.calc_gen_loss(pred_full_grad)
                    losses_gail.append(loss_gail.item()/B)   
                    
                    
                    loss_all = 1 * loss_recon + 0.002 * loss_gail
                    losses_all.append(loss_all.item()/B)
                    loss_sum = loss_sum + loss_all

                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()
                losses_sum = loss_sum.item() / B
                loss_train += loss_sum.item() * batch_size

            if FineTune:

                net_pred.eval()
                selector.train()
                with torch.no_grad():
                    data_out,results, other_loss, attention_logits_gt  , attention_weights_gt, ip_results, att_feats , l2_norm= net_pred(x_c, y_c,epo)
                    # torch.cuda.empty_cache()
                    if CSV :
                        B, P, J, T = ip_results[0].shape
                        # Prepare a list to hold row dictionaries
                        data = []
                        for b in range(B):
                            for p in range(P):
                                for k in range(len(ip_results)):
                                    for t in range(0,25,1):  
                                        data.append({
                                            'batch': b,
                                            'person': p,
                                            'level': k,
                                            'Timestep' : t,
                                            'ipscore': ip_results[k][b, p, :, t+25].mean().cpu().numpy(),
                                            'Weight':W_A_scores[b,p,t,:,k,0].mean().cpu().numpy(),
                                            "Amp": W_A_scores[b,p,t,:,k,1].mean().cpu().numpy(),
                                            "Att_feats" : att_feats[k][b,p,:,t+25].mean().cpu().numpy(),
                                            "norm" :l2_norm[b,p,t,k].cpu().numpy(),
                                            'choosed' : "Yes" if attention_weights_gt[b,p,t,k] >0.9 else "No"
                                        })
                        df = pd.DataFrame(data)

                        # Save to CSV
                        df.to_csv('ip_scores_data.csv', index=False)

                        return
                results = torch.stack(results, dim=0)         
                for i in range(1):
                    
                    result_x = results[:,(64 * i):(64*(i+1)),...]
                    y_c_x = y_c[(64 * i):(64*(i+1))]
                    # net_pred_x=net_pred.dct.detach()
                    data_out_x = data_out[(64 * i):(64*(i+1))]
                    attention_logits_gt_x = attention_logits_gt[(64 * i):(64*(i+1))].detach()
                    # W_A_scores_x =W_A_scores[(64 * i):(64*(i+1))].cuda().detach()
                    attention_weights_gt_x=attention_weights_gt[(64 * i):(64*(i+1))].detach()
                    ip_x =torch.stack(ip_results)[:,(64 * i):(64*(i+1)),...].detach()
                    Att_feats_x=torch.stack(att_feats)[:,(64 * i):(64*(i+1)),...].detach()
                    l2_norm_x = l2_norm[(64 * i):(64*(i+1))]
                    predic,selector_loss = selector(result_x, y_c_x,net_pred.dct.detach(), epo, predic_gt = data_out_x, teacher_logits=attention_logits_gt_x , teacher_weights=attention_weights_gt_x, ip = ip_x, Att_feats = Att_feats_x, l2_error = l2_norm_x)
                    optimizer_s.zero_grad()
                    selector_loss.backward()
                    optimizer_s.step()

            
    
        res_dic = {"loss_train" : loss_train / n }
        return res_dic

    else: #test
        net_pred.eval()
        adaptive_selector.eval()
        mpjpe_joi = np.zeros([opt.seq_len])
        ape_joi = np.zeros([5])
        vim_joi = np.zeros([5])
        # n = 0
        for batch_idx, (x, y) in enumerate(data_loader): # raw_in_n + out_n
            if np.shape(x)[0] < opt.batch_size:
                continue #when only one sample in this batch
            n += batch_size

            x = x.float().cuda()
            y = y.float().cuda()
            # print(x.shape, y.shape)

            data_out, _ = net_pred(x, y,-1)#[:,:,0]  # bz, 2kz, 108
            data_gt = y.transpose(2, 3)
            num_per = y.shape[1]


            data_gt = data_gt.reshape(opt.batch_size, num_per, opt.seq_len, nb_kpts, 3)
            data_out = data_out.reshape(opt.batch_size, num_per, opt.seq_len, nb_kpts, 3)
            tmp_joi = torch.sum(torch.mean(torch.mean(torch.norm(data_gt - data_out, dim=4), dim=3), dim=1), dim=0)
            # print(tmp_joi)
            mpjpe_joi += tmp_joi.cpu().data.numpy()

            tmp_ape_joi = APE(data_out[:, :, opt.frame_in:, :, :], data_gt[:, :, opt.frame_in:, :, :], [4, 9, 14, 19, 24])
            ape_joi += tmp_ape_joi#.data.numpy()

            data_vim_gt = data_gt[:, :, opt.frame_in:, :, :].transpose(2, 1)
            data_vim_gt = data_vim_gt.reshape(opt.batch_size, opt.seq_len, -1, 3)
            data_vim_pred = data_out[:, :, opt.frame_in:, :, :].transpose(2, 1)
            data_vim_pred = data_vim_pred.reshape(opt.batch_size, opt.seq_len, -1, 3)
            tmp_vim_joi = batch_VIM(data_vim_gt.cpu().data.numpy(), data_vim_pred.cpu().data.numpy(), [4, 9, 14, 19, 24])
            vim_joi += tmp_vim_joi#.data.numpy()



        mpjpe_joi = mpjpe_joi/n * 1000  # n = testing dataset length
        ape_joi = ape_joi/n * 1000 * opt.batch_size
        vim_joi = vim_joi/n * 100
        # print(ape_joi.shape, vim_joi.shape)
        select_frame = [4, 9, 14, 19, 24]
        print(mpjpe_joi[opt.frame_in:][select_frame])
        print("APE: ", ape_joi)
        print("VIM: ", vim_joi)

        mpjpe_mean = np.mean(mpjpe_joi[opt.frame_in:][select_frame])
        mpjpe_avg = np.mean(mpjpe_joi[opt.frame_in:])
        ape_mean = np.mean(ape_joi)
        vim_mean = np.mean(vim_joi)

        res_dic = {"mpjpe_joi": mpjpe_joi}


        if opt.save_results:
            import json
            key_exp = opt.exp + '_testepo'+str(opt.test_epoch)
            print('save name exp:', opt.exp)
            print('MPJPE mean: ', mpjpe_mean)
            print('MPJPE AVG: ', mpjpe_avg)
            print('APE_mean: ', ape_mean)
            print('VIM_mean: ', vim_mean)

            ts = "AGV"

            results = {key_exp: {}}
            results[key_exp][ts]={"mpjpe_joi": mpjpe_joi.tolist()}

            with open('{}/results.json'.format(opt.ckpt), 'w') as w:
                json.dump(results, w)

        return res_dic

def eval(opt, net_pred, data_loader, nb_kpts, epo ,selector = None, FineTune = False):
    net_pred.eval()
    if selector:
        selector.eval()
    mpjpe_joi = np.zeros([opt.seq_len])
    ape_joi = np.zeros([5])
    vim_joi = np.zeros([5])
    n = 0
    eval_frame = [5,10,15,20,25]
    loss_list1={}
    aligned_loss_list1 = {}
    mean_output = 0
    W_A_scores_out = torch.zeros((opt.batch_size,5,25,45,opt.k_level+1,2)).cuda()
    for batch_idx, (x, y) in enumerate(data_loader): # in_n + kz
        torch.cuda.empty_cache()
        print("eval batch",batch_idx)
        if np.shape(x)[0] < opt.batch_size:
            continue #when only one sample in this batch
        n += opt.batch_size

        x = x.float().cuda()
        y = y.float().cuda()

        with torch.no_grad():
            if not FineTune:
                data_out,results , attn_weights, W_A_scores ,mean_Of_levels,_,_= net_pred(x, y, epo,Train=False, FineTune = False) 
            else:
                _, results , attn_weights, W_A_scores, _,ip_results, att_feats= net_pred(x, y, epo,Train=False)
                results = torch.stack(results, dim=0)
                data_out, mean_Of_levels = selector(results, y,net_pred.dct,epo, Feats= W_A_scores.cuda(),Train=False,ip = torch.stack(ip_results).detach(), Att_feats = torch.stack(att_feats).detach()  )  
            # data_out,results, other_loss = net_pred(x, y, -1)

        # data_out, loss = net_pred(x, y,epo)#[:,:,0]  # bz, 2kz, 108
        num_per = y.shape[1]
        data_gt = y.transpose(2, 3)
        data_gt = data_gt.reshape(opt.batch_size, num_per, opt.seq_len, nb_kpts//3, 3)
        data_out = data_out.reshape(opt.batch_size, num_per, opt.seq_len, nb_kpts//3, 3)
        mean_output = mean_output + mean_Of_levels
        for j in eval_frame:
            mpjpe=torch.sqrt(((data_gt[:,:, opt.frame_in:opt.frame_in+j, :, :] - data_out[:,:, opt.frame_in:opt.frame_in+j, :, :]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).mean(dim = -1).mean(dim = -1).cpu().data.numpy().tolist()
            aligned_loss=torch.sqrt(((data_gt[:,:, opt.frame_in:opt.frame_in+j, :, :] - data_gt[:,:, opt.frame_in:opt.frame_in+j, 0:1, :] - data_out[:,:, opt.frame_in:opt.frame_in+j, :, :] + data_out[:,:, opt.frame_in:opt.frame_in+j, 0:1, :]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).mean(dim = -1).mean(dim = -1).cpu().data.numpy().tolist()           
            # root_loss=torch.sqrt(((prediction_1[:, :j, 0, :] - gt_1[:, :j, 0, :]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().data.numpy().tolist()
            if j not in loss_list1.keys():
                loss_list1[j] = mpjpe
                aligned_loss_list1[j] = aligned_loss
                # root_loss_list1[j] = []
            else:        
                loss_list1[j] += mpjpe
                aligned_loss_list1[j] += aligned_loss
            # root_loss_list1[j].append(np.mean(root_loss))
        W_A_scores_out +=W_A_scores.cuda()
    stats = {}
    no_batches = n/opt.batch_size
    W_A_scores_out = W_A_scores_out/no_batches
    W_A_scores_out = torch.norm(W_A_scores_out,dim=3)
    for j in eval_frame:
        e1, e2 = loss_list1[j]/no_batches*1000, aligned_loss_list1[j]/no_batches*1000
        prefix = 'val/frame%d/' % j
        stats[prefix + 'err'] = e1
        stats[prefix + 'err aligned'] = e2
        stats["MeanLevels"] = mean_output/no_batches
    if Viz_scores:

        print("the shape of ",W_A_scores_out .shape)
        all_images = []
        for b in range(opt.batch_size):
            for k in range(opt.k_level+1):
                data = attn_weights[b, :, :, k]  # (P, T)

                fig, ax = plt.subplots()
                im = ax.imshow(data.cpu(), cmap="viridis", aspect="auto")

                # Annotate each cell with its value
                for i in range(5):
                    for j in range(opt.frame_in):
                        value_1 = W_A_scores_out[0,i, j,k,0]
                        value_2 = W_A_scores_out[0,i, j,k,1]
                        ax.text(j, i, f"{value_1:.2f}\n{value_2:.2f}", 
        ha="center", va="center", color="white", fontsize=12)

                ax.set_title(f"Batch {b} - Class {k}")
                ax.set_xlabel("T (Time)")
                ax.set_ylabel("P (Points)")
                fig.colorbar(im, ax=ax)
                all_images.append(wandb.Image(fig, caption=f"B{b} C{k}"))
                plt.close(fig)
        wandb.log({"Annotated Heatmaps": all_images})
        
        # stats[prefix + 'err root'] = e3
    if epo >= 0:
        stats['epoch'] = epo
        wandb.log(stats)
    else:
        pprint(stats)
    return e1, e2
def APE(V_pred, V_trgt, frame_idx):

    V_pred = V_pred - V_pred[:, :, :, 0:1, :]
    V_trgt = V_trgt - V_trgt[:, :, :, 0:1, :]

    err = np.arange(len(frame_idx), dtype=np.float_)

    for idx in range(len(frame_idx)):
        err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2),dim=1).cpu().data.numpy().mean()
    return err

def batch_VIM(GT, pred, select_frames):
    '''Calculate the VIM at selected timestamps.

    Args:
        GT: [B, T, J, 3].

    Returns:
        errorPose: [T].
    '''
    errorPose = np.power(GT - pred, 2)
    errorPose = np.sum(errorPose, axis=(2, 3))
    errorPose = np.sqrt(errorPose)
    errorPose = errorPose.sum(axis=0)
    # scale = 100
    return errorPose[select_frames]# * scale

if __name__ == '__main__':
    option = Options().parse()

    wandb_args = {"expname" : "IAFormer",
                "data" : "Wusi",
                "epochs" : "80",
                "save_freq": 10,
                "batch_size": "96",
                "seed": "1234567890",
                "k_levels": 3,

                "lr": option.lr_now,
                "lr_decay": option.lr_decay_rate,
                "dropout": option.drop_out,

                "d_model": option.d_model}
    main(option,wandb_args)


