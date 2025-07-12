import torch
import torch.optim as optim
import numpy as np
import torch_dct as dct #https://github.com/zh217/torch-dct
import time

from model.IAFormer import IAFormer
from utils import other_utils
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import os
import yaml
from pprint import pprint
from easydict import EasyDict
from tqdm import tqdm
from torchviz import make_dot

from option.option_Mocap import Options

from utils.dataset import MPMotion

# import ipdb
import argparse
import wandb
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def set_random_seed(seed):
    # random.seed(seed)
    # #np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.cuda.set_device(device=0)
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(seed)
    torch.manual_seed(seed)   
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True

    select_cuda = 0
    torch.cuda.set_device(device=select_cuda)
    print("The using GPU is device {0}".format(select_cuda))
    torch.autograd.set_detect_anomaly(True)


def main(opt,wandb_args):

    # opts = parse_args()
    # with open(opts.config) as f:
    #     args = yaml.load(f, Loader=yaml.Loader)

    set_random_seed(opt.seed)
    device = 'cuda'

    dataset = MPMotion(opt.train_data, concat_last=False)
    test_dataset = MPMotion(opt.test_data, concat_last=False)

     # Model Intializtion 

    nb_kpts = int(opt.in_features) #keypoints in 3d (joints number * 3)
    print('>>> MODEL >>>')
    model = IAFormer(seq_len=opt.seq_len, d_model=opt.d_model, opt=opt, num_kpt=nb_kpts, dataset=opt.dataset, k_levels = opt.k_level)
    model.cuda()
    lr_now = opt.lr_now
    start_epoch = 1
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    if opt.mode == "train":
        wandb.init(
                project="COG_Transformer_Wusi",
                config=wandb_args,
                name="test can this be IAformer"#"COG_lvl0_ Weighted_loss(joint+time) (not alignedV2)"#"COG_V4_lvl1_ Weighted_loss (alignedV2 (1, 0.9))(1,0.1,1))"
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,num_workers=0, pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True,num_workers=0, pin_memory=True)
        params = [
            {"params": model.parameters(), "lr": opt.lr_now}
        ]
        # optimizer = optim.Adam(params)
        # params_d = [
        #     {"params": discriminator.parameters(), "lr": args.lr}
        # ]
        # optimizer_d = optim.Adam(params_d)
        #optimizer = optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr_now,weight_decay=1e-2)
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr_now)

        min_error = 100000
        for epoch in tqdm(range(opt.epochs)):    
            print('Training epoch', epoch)
            print("Model is in training mode")


            ret_train = run_model(nb_kpts, model, opt.batch_size, optimizer, data_loader = dataloader, opt=opt, epo=epoch)
            mpjpe_mean,ape_mean = eval(opt, model, test_dataloader, nb_kpts, epoch)


            print(f"Train loss: {ret_train['loss_train']}")
            print(f"Evaulation MPE: {mpjpe_mean}")
            print(f"Evaulation APE: {ape_mean}")

            # mpjpe_mean, mpjpe_avg, ape_mean, vim_mean = eval(opt, model, test_dataloader, nb_kpts, epo)

            # discriminator.train()
            lr_now = other_utils.lr_decay_mine(optimizer, lr_now,  0.1 ** (1 / 50))

def run_model(nb_kpts, net_pred, batch_size, optimizer=None, data_loader=None, opt=None, epo=0):

    n = 0
    net_pred.train()
    loss_train = 0
    for batch_idx, (input_seq, output_seq) in enumerate(data_loader): # input sequence , output sequence =x , y

        torch.cuda.empty_cache()
        B = input_seq.shape[0]
        if np.shape(input_seq)[0] < opt.batch_size:
            continue #when only one sample in this batch
        n += batch_size

        input_seq = input_seq.float().cuda()
        output_seq = output_seq.float().cuda()

        input_seq_c = input_seq.clone().detach()
        output_seq_c = output_seq.clone().detach()
        data_out, mix_loss = net_pred(input_seq_c, output_seq_c,batch_idx)

        data_gt = output_seq_c.transpose(2, 3)
        loss = mix_loss

        optimizer.zero_grad()
        loss.backward()
        # dot = make_dot(loss, params=dict(net_pred.named_parameters()))
        # dot.render("png/computation_graph", format="png", view=True)
        loss_train += loss.item() * batch_size
        optimizer.step()

    res_dic = {"loss_train" : loss_train / n }
    return res_dic

def eval(opt, net_pred, data_loader, nb_kpts, epo):
    print("start eval")
    net_pred.eval()
    # mpjpe_joi = np.zeros([opt.frame_out +1])
    # ape_joi = np.zeros([5])
    # vim_joi = np.zeros([5])
    n = 0
    eval_frame = [5,10,15,20,25]
    loss_list1={}
    aligned_loss_list1 = {}
    # root_loss_list1={}
    for batch_idx, (input_seq, output_seq) in enumerate(data_loader):
    # for (input_seq, output_seq) in tqdm(data_loader): # in_n + kz
        if np.shape(input_seq)[0] < opt.batch_size:
            continue #when only one sample in this batch
        # n += 1
        n += opt.batch_size
        input_seq = input_seq.float().cuda()
        output_seq = output_seq.float().cuda()
        
        data_out, loss = net_pred(input_seq, output_seq,-1)#[:,:,0]  # bz, 2kz, 108

        num_per = output_seq.shape[1]
        data_gt = output_seq.transpose(2, 3)
        B = data_gt.shape[0]
        # print("this is the shape of ground truth ",data_gt.shape)

        print(data_out.shape, data_gt.shape)
        data_gt = data_gt.reshape(opt.batch_size,num_per, opt.seq_len, nb_kpts//3, 3)
        data_out = data_out.reshape(opt.batch_size,num_per, opt.seq_len, nb_kpts//3, 3)
        for j in eval_frame:
            mpjpe=torch.sqrt(((data_gt[:,:, opt.frame_in:opt.frame_in+j, :, :] - data_out[:,:, opt.frame_in:opt.frame_in+j, :, :]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).mean(dim = -1).sum(dim = -1).cpu().data.numpy().tolist()
            aligned_loss=torch.sqrt(((data_gt[:,:, opt.frame_in:opt.frame_in+j, :, :] - data_gt[:,:, opt.frame_in:opt.frame_in+j, 0:1, :] - data_out[:,:, opt.frame_in:opt.frame_in+j, :, :] + data_out[:,:, opt.frame_in:opt.frame_in+j, 0:1, :]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).mean(dim = -1).sum(dim = -1).cpu().data.numpy().tolist()           
            # root_loss=torch.sqrt(((prediction_1[:, :j, 0, :] - gt_1[:, :j, 0, :]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().data.numpy().tolist()
            if j not in loss_list1.keys():
                loss_list1[j] = mpjpe
                aligned_loss_list1[j] = aligned_loss
                # root_loss_list1[j] = []
            else:        
                loss_list1[j] += mpjpe
                aligned_loss_list1[j] += aligned_loss
            # root_loss_list1[j].append(np.mean(root_loss))

        stats = {}
        for j in eval_frame:
            e1, e2 = loss_list1[j]/n*1000, aligned_loss_list1[j]/n*1000
            prefix = 'val/frame%d/' % j
            stats[prefix + 'err'] = e1
            stats[prefix + 'err aligned'] = e2
            # stats[prefix + 'err root'] = e3
        if epo >= 0:
            stats['epoch'] = epo
            wandb.log(stats)
        else:
            pprint(stats)
        return e1, e2

        # kofta=torch.sqrt(((data_gt - data_out) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().data.numpy().tolist()
        # print(f"this is the other one kofta mean {np.mean(kofta)}")
        # tmp_joi = torch.sum(torch.mean(torch.mean(torch.norm(data_gt - data_out, dim=4), dim=3), dim=1), dim=0)
        # # print(tmp_joi)
        # mpjpe_joi += tmp_joi.cpu().data.numpy()

        # tmp_ape_joi = APE(data_out[:, :, :, :, :], data_gt[:, :, :, :, :], [4, 9, 14, 19, 24])
        # ape_joi += tmp_ape_joi#.data.numpy()

        # data_vim_gt = data_gt[:, :, opt.frame_in:, :, :].transpose(2, 1) 
        # data_vim_gt = data_vim_gt.reshape(opt.batch_size, opt.seq_len -opt.frame_in , -1, 3) 
        # data_vim_pred = data_out[:, :, opt.frame_in:, :, :].transpose(2, 1)
        # data_vim_pred = data_vim_pred.reshape(opt.batch_size, opt.seq_len-opt.frame_in, -1, 3)
        # tmp_vim_joi = batch_VIM(data_vim_gt.cpu().data.numpy(), data_vim_pred.cpu().data.numpy(), [4, 9, 14, 19, 24])
        # vim_joi += tmp_vim_joi#.data.numpy()
    # print(f" this is the numbr of batches: {n}")
    # mpjpe_joi = mpjpe_joi/n * 1000  # n = testing dataset length
    # ape_joi = ape_joi/n * 1000 * opt.batch_size
    # # vim_joi = vim_joi/n * 1000
    # print("APE shape",ape_joi.shape)
    # print("MPE shape",mpjpe_joi)
    # print("APE: ", ape_joi)
    # print("VIM: ", vim_joi)
    # select_frame = [4, 9, 14, 19, 24]
    # mpjpe_mean = np.mean(mpjpe_joi[:][select_frame])
    # # mpjpe_avg = np.mean(mpjpe_joi)
    # ape_mean = np.mean(ape_joi)
    # vim_mean = np.mean(vim_joi)


    # if opt.save_results:
    #     import json
    #     key_exp = 'epoch:'+str(epo)

    #     results = {key_exp: {}}
    #     results[key_exp]["mpjpe_joi"]=mpjpe_joi.tolist()
    #     results[key_exp]["mpjpe_mean"]=mpjpe_mean.tolist()
    #     results[key_exp]["ape_joi"]=ape_joi.tolist()
    #     results[key_exp]["ape_mean"]=ape_mean.tolist()
    #     results[key_exp]["vim_joi"]=vim_joi.tolist()
    #     results[key_exp]["vim_mean"]=vim_mean.tolist()
    #     # print(mpjpe_joi)
    #     with open('{}/eval_results.json'.format(opt.ckpt), 'a') as w:
    #         json.dump(results, w)
    #         w.write('\n')

    return mpjpe_mean, ape_mean#, vim_mean,mpjpe_avg

def APE(V_pred, V_trgt, frame_idx):

    V_pred = V_pred - V_pred[:, :, :, 0:1, :]
    V_trgt = V_trgt - V_trgt[:, :, :, 0:1, :]

    err = np.arange(len(frame_idx), dtype=np.float_)

    for idx in range(len(frame_idx)):
        err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2),dim=1).cpu().data.numpy().mean()
    return err

if __name__ == '__main__':
    option = Options().parse()
    wandb_args = {"expname" : option.expname,
                "data" : option.dataset,
                "epochs" : option.epochs,
                "save_freq": 10,
                "batch_size": option.batch_size,
                "seed": option.seed,
                "k_levels": option.k_level,

                "lr": option.lr_now,
                "lr_decay": option.lr_decay_rate,
                "dropout": option.drop_out, 

                "d_model": option.d_model}
                # "d_inner_g": 1024,
                # "d_inner_d": 1024,
                # "d_hidden": 512,

                # "lambda_gail": 0.002,
                # "lambda_recon": 1,

                # "gail_sample": True,
                # "share_d": True}

    # with open("wandb.yaml", "w") as yaml_file:
    #     yaml.dump(yaml_data, yaml_file, default_flow_style=False, sort_keys=False)             
    main(option, wandb_args)           
            
