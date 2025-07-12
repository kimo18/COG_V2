import torch
import pandas as pd
import numpy as np
class FeatsExtractor:
    def __init__(self,opt):
        self. opt = opt
        self.BONES = [
            (0, 1), (0, 4),         # root → hips
            (0, 8), (8, 7),         # root → neck → head
            (1, 2), (2, 3),         # left leg
            (4, 5), (5, 6),         # right leg
            (8, 9), (9, 10), (10, 11),     # left arm
            (8, 12), (12, 13), (13, 14)    # right arm
        ]

        # Symmetric joints (right ↔ left)
        self.SYMMETRIC_JOINTS = [
            (1, 4),  # hip left ↔ hip right
            (2, 5),  # knee
            (3, 6),  # ankle
            (9, 12),  # shoulder
            (10, 13),  # elbow
            (11, 14)   # wrist
        ]

        # Joint angle triplets (shoulder-elbow-wrist, hip-knee-ankle)
        self.ANGLE_TRIPLETS = [
            (9, 10, 11),     # left arm
            (12, 13, 14),    # right arm
            (1, 2, 3),       # left leg
            (4, 5, 6)        # right leg
        ]


    def compute_bone_lengths(self, pose):
        return torch.stack([
            (pose[:, j1] - pose[:, j2]).norm(dim=-1)
            for j1, j2 in self.BONES
        ], dim=1)  # (T, num_bones)

    def compute_joint_angles(self, pose):
        angles = []
        for a, b, c in self.ANGLE_TRIPLETS:
            v1 = pose[:, a] - pose[:, b]
            v2 = pose[:, c] - pose[:, b]
            v1 = v1 / v1.norm(dim=-1, keepdim=True)
            v2 = v2 / v2.norm(dim=-1, keepdim=True)
            dot = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
            angles.append(torch.acos(dot) * (180.0 / torch.pi))
        return torch.stack(angles, dim=1)  # (T, num_angles)

    def compute_symmetry_error(self, pose):
        return torch.stack([
            (pose[:, r] - pose[:, l]).norm(dim=-1)
            for r, l in self.SYMMETRIC_JOINTS
        ], dim=1)  # (T, num_symmetric_pairs)

    def compute_velocity(self, pose):
        v = torch.zeros_like(pose)
        v[1:] = (pose[1:] - pose[:-1])
        return v.norm(dim=-1)  # (T, J)

    def get_feats(self, predictions):
        # B, P, T, J = predictions.shape
        # level = predictions[:,:,self.opt.frame_out:,:].reshape(B,P,self.opt.frame_out,J//3,3)
        """
        predictions: List of tensors [(B, P, T, J, 3), ...] for each level
        Returns: List of tensors [(B, P, T, F), ...] per level
        """
        all_levels = []

        for level in predictions:
            B, P, T, J = level.shape
            level = level[:,:,self.opt.frame_out:,:].reshape(B,P,self.opt.frame_out,J//3,3)
            B, P, T, J,C = level.shape
            feats = []

            for b in range(B):
                person_feats = []
                for p in range(P):
                    pose = level[b, p]  # (T, J, 3)
                    bones = self.compute_bone_lengths(pose)  # (T, num_bones)
                    angles = self.compute_joint_angles(pose)  # (T, num_angles)
                    symmetry = self.compute_symmetry_error(pose)  # (T, num_symmetric_pairs)
                    velocity = self.compute_velocity(pose)  # (T, J)

                    full_feat = torch.cat([bones, angles, symmetry, velocity], dim=1)  # (T, F)
                    person_feats.append(full_feat)
                feats.append(torch.stack(person_feats, dim=0))  # (P, T, F)
            all_levels.append(torch.stack(feats, dim=0))  # (B, P, T, F)

        print(torch.stack(all_levels,dim = 3).shape)    
        # quit()
        return torch.stack(all_levels,dim = 3)  # List of (B, P, T, F)

    # def get_feats(self, pose):
    #     pose = pose[:,:,self.opt.frame_out:]
    #     f1 = self.compute_bone_lengths(pose)
    #     f2 = self.compute_joint_angles(pose)
    #     f3 = self.temporal_smoothness(pose)
    #     f4 = self.symmetry_error(pose)
    #     # print(f1.shape, f2.shape, f3.shape,f4.shape)
        

    #     out_feat = torch.cat([f1,f2,f3,f4],dim=-1)
    #     return out_feat
      



    # def compute_bone_lengths(self,pose):
    #     B,P,T,J = pose.shape
    #     pose = pose.reshape(-1,pose.shape[-1]//3,3)
          
    #     return torch.stack([
    #         (pose[:, j1] - pose[:, j2]).norm(dim=-1)
    #         for j1, j2 in self.BONES
    #     ], dim=1).reshape(B,P,T,len(self.BONES))
    #     # print(fun.shape)


    #     # array = tensor.cpu().numpy()

    #     # # Reshape to 2D if needed (e.g., T x (J*3))
    #     # flat_array = array.reshape(array.shape[0], -1)  # shape: (T, 45)

    #     # Convert to DataFrame
    #     # df = pd.DataFrame(fun.cpu().numpy())
    #     # quit()

        

    # def compute_joint_angles(self,pose):
    #     B,P,T,J = pose.shape
    #     pose = pose.reshape(-1,pose.shape[-1]//3,3)
    #     angles = []
    #     for a, b, c in self.ANGLE_TRIPLETS:
    #         v1 = pose[:, a] - pose[:, b]
    #         v2 = pose[:, c] - pose[:, b]
    #         v1 = v1 / v1.norm(dim=-1, keepdim=True)
    #         v2 = v2 / v2.norm(dim=-1, keepdim=True)
    #         dot = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
    #         angles.append(torch.acos(dot) * (180.0 / torch.pi))
    #     return torch.stack(angles, dim=1).reshape(B,P,T,len(self.ANGLE_TRIPLETS))

    # def temporal_smoothness(self,pose):
    #     B,P,T,J = pose.shape
    #     pose = pose.reshape(-1,pose.shape[-1]//3,3)
    #     v = (pose[:,1:] - pose[:,:-1])
    #     a = (v[:,1:] - v[:,:-1])
    #     j = (a[:,1:] - a[:,:-1])
    #     # print(v.shape, a.shape, j.shape)

    #     return torch.cat([v.reshape(B,P,T,-1),a.reshape(B,P,T,-1),j.reshape(B,P,T,-1)],dim = -1)

    # def symmetry_error(self,pose):
    #     B,P,T,J = pose.shape
    #     pose = pose.reshape(-1,pose.shape[-1]//3,3)
    #     diffs = [(pose[:, r] - pose[:, l]).norm(dim=-1) for r, l in self.SYMMETRIC_JOINTS]
    #     diffs=torch.stack(diffs, dim=1)
    #     diffs=diffs.reshape(B*P*T,-1)
        
    #     return torch.mean(diffs, dim = -1).reshape(B,P,T,-1)

    # def pose_diversity(pose):
    #     return (pose[1:] - pose[:-1]).norm(dim=-1).mean().item()

    # def evaluate_pose_quality(pred_levels):
    #     """
    #     pred_levels: list of np.ndarray, each of shape (B, P, T, J, 3)
    #     returns: dict of metrics per level
    #     """
    #     results = {}

    #     for m_idx, pred in enumerate(pred_levels):
    #         B, P, T, J, _ = pred.shape
    #         level_results = {
    #             'bone_length_std': [],
    #             'angle_mean': [],
    #             'angle_range': [],
    #             'smoothness': [],
    #             'symmetry_error': [],
    #             'pose_diversity': []
    #         }

    #         for b in range(B):
    #             for p in range(P):
    #                 pose = pred[b, p]  # (T, J, 3)

    #                 # Bone Length Consistency
    #                 bone_lengths = compute_bone_lengths(pose)
    #                 level_results['bone_length_std'].append(bone_lengths.std())

    #                 # Joint Angles
    #                 angles = compute_joint_angles(pose)
    #                 level_results['angle_mean'].append(angles.mean())
    #                 level_results['angle_range'].append(angles.max() - angles.min())

    #                 # Temporal Smoothness
    #                 smooth = temporal_smoothness(pose)
    #                 level_results['smoothness'].append(smooth)

    #                 # Symmetry
    #                 symm = symmetry_error(pose)
    #                 level_results['symmetry_error'].append(symm)

    #                 # Pose Diversity
    #                 diversity = pose_diversity(pose)
    #                 level_results['pose_diversity'].append(diversity)

    #         # Aggregate statistics
    #         results[f'Level_{m_idx}'] = {
    #             'bone_length_std': np.mean(level_results['bone_length_std']),
    #             'angle_mean': np.mean(level_results['angle_mean']),
    #             'angle_range': np.mean(level_results['angle_range']),
    #             'velocity': np.mean([s['velocity'] for s in level_results['smoothness']]),
    #             'acceleration': np.mean([s['acceleration'] for s in level_results['smoothness']]),
    #             'jerk': np.mean([s['jerk'] for s in level_results['smoothness']]),
    #             'symmetry_error': np.mean(level_results['symmetry_error']),
    #             'pose_diversity': np.mean(level_results['pose_diversity'])
    #         }

    #     return results























































# import matplotlib.pyplot as plt
#             from mpl_toolkits.mplot3d import Axes3D

#             # Example 3D pose data: (15 joints, each with x, y, z)
#             pose_3d_x = x_c[0,:,:,0].reshape(-1,15,3).cpu().numpy()  # Replace this with your actual pose data
            # print(pose_3d.shape)
            # Optional: Define skeleton connections (pairs of joint indices)






            # skeleton = [
            #     (0, 1),(0, 4),    # root ->pelvis
            #     (0, 8), (8,7),   # root->neck->head
            #     (1, 2), (2, 3),        # leg1
            #     (4, 5), (5, 6),        # leg2
                      
            #     (8, 9), (9,10), (10, 11),   # arm1
            #     (8, 12), (12, 13),(13,14)              # arm2
            # ]

            # # Plotting
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # # Plot joints
            # for p in range(5):
            #     pose_3d = pose_3d_x[p]
            #     ax.scatter(pose_3d[:, 0],pose_3d[:, 1], pose_3d[:, 2], c='r', s=40)

            #     # Plot bones
            #     for joint_start, joint_end in skeleton:
            #         xs = [pose_3d[joint_start, 0], pose_3d[joint_end, 0]]
            #         ys = [pose_3d[joint_start, 1], pose_3d[joint_end, 1]]
            #         zs = [pose_3d[joint_start, 2], pose_3d[joint_end, 2]]
            #         ax.plot(xs, ys, zs, c='b')

            #     # Adjust view
            #     ax.set_xlabel('X')
            #     ax.set_ylabel('Y')
            #     ax.set_zlabel('Z')
            #     ax.view_init(elev=20, azim=60)  # Customize camera angle

            # plt.savefig('fun.png')
            # return
