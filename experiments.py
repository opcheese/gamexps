import main
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import simple_model
import model
import tqdm

if __name__ == "__main__":
    game = main.MNKGame(width=3,height=3,n_in_row=3 )
    state = main.MNKState(game)

    st ="""x..
o..
o.x"""
    state = main.MNKState.emulate_state(game,st)

    #pvn = simple_model.PolicyValueNet(3,3,model_file="simpletext400.model")
    pvn = model.UnetPolicyValueNet(3,3,fmodel_file="unetsimplenet_-1.fmodel")
    
    print(state.current_player())
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


#     st = """xox
# xoo
# .x."""
#     state = main.MNKState.emulate_state(game,st)