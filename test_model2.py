import main
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from itertools import cycle, islice
import random
import model
import tqdm
import wandb


wandb.init(project="new-tic-tac")
solved = {}


class CustomGameDataset(Dataset):
    def __init__(self, file_paths,batch_size):
        self.file_paths = file_paths
        self.Xs = []
        for X_path in file_paths:
            X = np.load(X_path, mmap_mode='r')
            self.Xs.append(X)
        self.batch_size = batch_size
        self.sz = (self.batch_size,) + self.Xs[0][0][['obs','mask','value']].shape
        self.minibatch = int(batch_size/len(file_paths))
        # self.files = []
        # for path in file_paths:
        #     X = np.load(path, mmap_mode='r')
        #     self.files.append(X)

    def __len__(self):
        lens = map(len,self.Xs)
        res = min(lens)-self.batch_size
        return res

    def __getitem__(self,idx):
        res1 = torch.empty(size=(self.batch_size,3,4,4)).type(torch.cuda.FloatTensor)
        res2 = torch.empty(size=(self.batch_size,16)).type(torch.cuda.FloatTensor)
        res3 = torch.empty(size=(self.batch_size,16)).type(torch.cuda.FloatTensor)

        res4 = torch.empty(size=(self.batch_size,1)).type(torch.cuda.FloatTensor)

        ind = -1
        for x_i in self.Xs :
            X = x_i
            for i in range(idx,idx+self.minibatch):
                ind+=1
                line = X[i]
                res1_t = torch.from_numpy(line['obs']).type(torch.cuda.FloatTensor).cuda().type(torch.cuda.FloatTensor)
                res1[ind] = res1_t
                res2_t = torch.from_numpy(line['policy']).type(torch.cuda.FloatTensor).cuda().type(torch.cuda.FloatTensor)
                res2[ind] = res2_t
                res3_t = torch.from_numpy(line['mask']).type(torch.cuda.FloatTensor).cuda().type(torch.cuda.FloatTensor)
                res3[ind] = res3_t
                res4_t = torch.as_tensor(line['value']).type(torch.cuda.FloatTensor).cuda().type(torch.cuda.FloatTensor)
                res4[ind] = res4_t
            idx = random.randint(0,self.__len__()-1)
            # if res1==None:
            #     res1 = torch.from_numpy(line['obs'])
            # else:
            #     res1_t = torch.from_numpy(line['obs'])
            #     if i == 1:
                
            #         res1 = torch.stack((res1,res1_t))
            #     else:
            #         res1_t = torch.expand_dims(res1_t)
            #         res1 = torch.stack((res1,res1_t))

            # if res2==None:
            #     res2 = torch.from_numpy(line['mask'])
            # else:
            #     res2_t = torch.from_numpy(line['mask'])
            #     res2 = torch.stack((res2,res2_t))
            #     b = 213
            # if res3==None:
            #     res3 = torch.as_tensor(line['value'])
            # else:
            #     res3_t = torch.as_tensor(line['value'])
            #     res3 = torch.stack((res3,res3_t))
            # res1 = torch.from_numpy(line['obs'])
            # c = torch.from_numpy(line['mask'])
            # d = torch.as_tensor (line['value'])
            # bc = torch.cat ((b,c))
            # bcd = torch.cat((bc,d))
            # res[ind] = bcd
            
        return res1,res2,res3,res4

if __name__ == "__main__":
    file_names = []
    for i in range(1):
        file_names.append("np444_{}.npy".format(i))
    iterable_dataset = CustomGameDataset(file_names,256)
    loader = DataLoader(iterable_dataset, batch_size=None)
    game = main.MNKGame(width=4,height=4,n_in_row=4 )
    state = main.MNKState(game)
    modelNetwork = model.Net(4,4).cuda()
    pvn = model.UnetPolicyValueNet(board_width=4,board_height=4, model=modelNetwork)

    ##wandb.watch(pvn.policy_value_net)

    epoch = 1
    pvn.save_model('simpletext'+str(0)+'.model')

    #pvn.load_model('simpletext.model')
    pvn.policy_value_net.train()
    for k in range(epoch):
      for i, (inputs,masks, target_policy, target_value) in  tqdm.tqdm(enumerate(loader)):
        # evaluate the model on the test set
      
        yhat = pvn.train_step(inputs,masks,target_policy,target_value,0.005)
        if i%1000==0: 
          wandb.log(yhat)
          print(yhat)
        if i%1000==0:


            st ="""x...
....
o.x.
o..x"""
            state = main.MNKState.emulate_state(game,st)
            tens = state.observation_tensor()
            print(tens)
            tens = np.expand_dims(tens, axis=0)
            t = torch.from_numpy(tens)
            t = t.to('cuda')
            t = t.type(torch.cuda.FloatTensor)

            msk = state.legal_actions_mask()
            m = torch.from_numpy(msk)
            m = m.to('cuda')
            m = m.type(torch.cuda.FloatTensor)
            # ndarray isn't hashable
            pvn.policy_value_net.eval()
            r = pvn.policy_value_net(t,m)
            pvn.policy_value_net.train()

            print(r)
        if i%10000==0:
            pvn.save_model('simpletext2'+str(i)+'.model')
    