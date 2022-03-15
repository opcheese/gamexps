import main
import numpy as np
import pickle

solved = {}

def solve_game(state,i):  
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
    values[action] = solve_game(state.child(action),i+1)
  value = values.max() #if max_player else values.min()
  
  best_actions = np.where((values == value) & (act_mask == 1))
  policy = np.zeros_like(act_mask)
  policy[best_actions[0][0]] = 1  # Choose the first for a deterministic policy.
  if max_player:
    solved[state_str] = (obs,act_mask, policy, value)
  else:
    solved[state_str] = (obs,act_mask, policy, value)
  if i==1:
    print(i)  
  return -value


if __name__ == "__main__":
    print(123)
    game = main.MNKGame(width=3,height=3,n_in_row=3 )
    state = main.MNKState(game)
#     st = """x..
# o..
# o.x"""
#     state = main.MNKState.emulate_state(game,st)
    solve_game(state,0)
    a = solved
    #exit()
    try: 
      a_file = open("data3.pkl", "wb")
      pickle.dump(solved, a_file)
    except:
      pass 