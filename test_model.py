import main
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import simple_model
import model
import tqdm
import requests
import wandb


wandb.init(project="new-tic-tac")
solved = {}

def solve_game(state):
  state_str = str(state)
  if state_str in solved:
    return solved[state_str][-1]
  if state.is_terminal(): 
      return -state.returns()[state.current_player_nt()]


  max_player = state.current_player() == 0
  obs = state.observation_tensor()
  act_mask = np.array(state.legal_actions_mask())
  values = np.full(act_mask.shape, -2) #if max_player else 2)
  for action in state.legal_actions():
    values[action] = solve_game(state.child(action))
  value = values.max() #if max_player else values.min()
  
  best_actions = np.where((values == value) & (act_mask == 1))
  policy = np.zeros_like(act_mask)
  policy[best_actions[0][0]] = 1  # Choose the first for a deterministic policy.
  if max_player:
    solved[state_str] = (obs,act_mask, policy, value)
  else:
    solved[state_str] = (obs,act_mask, policy, value)

  return -value

class CustomGameDataset(Dataset):
    def __init__(self, boards,masks,policies,values):
        self.boards = torch.from_numpy(boards)
        self.masks = torch.from_numpy(masks)

        self.policy = torch.from_numpy(policies)
        self.value = torch.from_numpy(values)
      

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        board = board.to('cuda')
        board = board.type(torch.cuda.FloatTensor)

        mask = self.masks[idx]
        mask = mask.to('cuda')
        mask = mask.type(torch.cuda.FloatTensor)
        
        policy = self.policy[idx]
        policy = policy.to('cuda')
        policy = policy.type(torch.cuda.FloatTensor)

        value = self.value[idx]
        value = value.to('cuda')
        value = value.type(torch.cuda.FloatTensor)

        return board,mask, policy,value

if __name__ == "__main__":
    print(123)
    game = main.MNKGame(width=3,height=3,n_in_row=3 )
    state = main.MNKState(game)
#     st = """x..
# o..
# o.x"""
#     state = main.MNKState.emulate_state(game,st)
    solve_game(state)
    a = solved
    #exit()
    # a_file = open("data.pkl", "wb")
    # pickle.dump(solved, a_file)
    # for s in solved:
    #     print(s)
    #     print(solved[s])
    obss = []
    masks = []
    policies = []
    values = []

    for s in solved:
      obs,mask,policy,value = solved[s]
      obss.append(obs)
      masks.append(mask)
      policies.append(policy)
      values.append(value)

    np_obss = np.array(obss)
    np_masks = np.array(masks)
    np_policies = np.array(policies)
    np_values = np.array(values)

    cds = CustomGameDataset(np_obss,np_masks,np_policies,np_values)
    trainLoader = torch.utils.data.DataLoader(dataset = cds, batch_size=128)

    #pvn = simple_model.PolicyValueNet(3,3)

    pvn = model.UnetPolicyValueNet(board_width=3,board_height=3)

    ##wandb.watch(pvn.policy_value_net)

    epoch = 5000
    pvn.save_model('simpletext'+str(0)+'.model')

    #pvn.load_model('simpletext.model')
    pvn.policy_value_net.train()
    for k in tqdm.tqdm(range(epoch)):
      for i, (inputs,masks, target_policy, target_value) in enumerate(trainLoader):
        # evaluate the model on the test set
      
        yhat = pvn.train_step(inputs,masks,target_policy,target_value,0.01)
        if i%100==0: 
          wandb.log(yhat)
      if k%100==0:


        st ="""x..
o..
o.x"""
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
        r = pvn.policy_value_net(t,m)

        print(r)

        pvn.save_model('simpletext'+str(k)+'.model')
      