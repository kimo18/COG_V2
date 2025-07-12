import torch.utils.data as data
import torch
import numpy as np

class MPMotion(data.Dataset):
    def __init__(self, data_path, in_len = 25, max_len = 50, concat_last = False, mode = "Train"):
        print(f'>>> DATA loading from path: {data_path}>>>')
        self.data = np.load(data_path, allow_pickle=True)
        print(f'>>> {mode}ing dataset Shape: {self.data.shape}') #Sequence,#person,#Frame,#keypoints
        self.len = len(self.data)
        self.max_len = max_len
        self.in_len = in_len
        self.diff = max_len-in_len
        # self.concat_last = concat_last
            
    def __getitem__(self, index):
        input_seq=self.data[index][:,:self.in_len,:]     
        output_seq=self.data[index][:,:self.max_len,:]
        pad_idx = np.repeat([self.in_len - 1], self.diff)
        i_idx = np.append(np.arange(0, self.in_len), pad_idx)
        input_seq = input_seq.transpose(0, 2, 1)
        input_seq = input_seq[:, :, i_idx]
        output_seq = output_seq.transpose(0, 2, 1)    
        return input_seq, output_seq
        
    def __len__(self):
        return self.len