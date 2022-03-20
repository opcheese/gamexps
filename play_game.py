import model
import main
import numpy as np
import torch
import evaluator
import mcts

pvn = model.UnetPolicyValueNet(board_width=4,board_height=4,model_file="simpletext230000.model")
game = main.MNKGame(width=4,height=4,n_in_row=4 )
state = main.MNKState(game)

az_evaluator = evaluator.AlphaZeroEvaluator(game, pvn)
bot = mcts.MCTSBot(
      game,
      2,
      300,
      az_evaluator,
      solve=True,
      dirichlet_noise=None,
      child_selection_fn=mcts.SearchNode.puct_value,
      verbose=True,
      dont_return_chance_node=True)

pvn.policy_value_net.eval()

while not state.is_terminal():
    print(state)
    num1 = int(input())
    state._apply_action(num1)
    if not state.is_terminal():
        tens = state.observation_tensor()
        #print(tens)
        tens = np.expand_dims(tens, axis=0)
        t = torch.from_numpy(tens)
        t = t.to('cuda')
        t = t.type(torch.cuda.FloatTensor)

        msk = state.legal_actions_mask()
        m = torch.from_numpy(msk)
        m = m.to('cuda')
        m = m.type(torch.cuda.FloatTensor)
        # ndarray isn't hashable
        #pol,val = pvn.policy_value_net(t,m)
        #p = pol.cpu().detach().numpy()
        #action = np.argmax(p)
        root = bot.mcts_search(state)

        policy = np.zeros(game.width*game.height)
        best_action = root.best_child().action
        state._apply_action(best_action)