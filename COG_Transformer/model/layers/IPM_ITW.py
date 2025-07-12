# Level Perceptron
import torch

def Trajectory_Weight(opt, skeleton,Viz = False):
    if not Viz:
        num_person = skeleton.shape[0]
        # print(skeleton.shape)  # [3, 45, 75]
        center_co = torch.mean(skeleton[:, :3, opt.frame_in-1], dim=0)
        # print(center_co.shape, center_co.unsqueeze(1).repeat(1, 50).shape, skeleton[:, :3, :50].shape)
        move_score = torch.ones(num_person)

        for i in range(num_person):
            move_score[i] = torch.norm(skeleton[i, :3, :opt.frame_in]-center_co.unsqueeze(1).repeat(1, opt.frame_in))

        lp_score = LP_score(move_score)
        return lp_score.cuda()
    else:
        num_person = skeleton.shape[0]
        center_co = torch.mean(skeleton[:, :3, :], dim=0)
        move_score = torch.ones((num_person,skeleton.shape[2]))

        for i in range(num_person):
            move_score[i] = torch.norm(skeleton[i, :3, :]-center_co, dim = 0 )
        lp_score = LP_score(move_score,Viz = True)
        return lp_score.cuda()


def LP_score(move_score, Viz = False):
    if not Viz:
        num_person = len(move_score)
        lp_score = torch.ones(num_person)

        for i in range(num_person):
            lp_score[i] = torch.log(move_score[i]/torch.sum(move_score)+1)
        
        return lp_score
    else:
        num_person = len(move_score)
        lp_score = torch.ones(move_score.shape)

        for i in range(num_person):
            lp_score[i] = torch.log(move_score[i]/torch.sum(move_score, dim = 0)+1)
        return lp_score      