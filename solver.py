import main
import numpy as np
import pickle



def solve_game(state,i, tosolve = []):  
  state_str = str(state)
  if state_str in solved:
    return -solved[state_str][-1]
  if state.is_terminal(): 
      return -state.returns()[state.current_player_nt()]


  max_player = state.current_player() == 0
  obs = state.observation_tensor()
  act_mask = np.array(state.legal_actions_mask())
  values = np.full(act_mask.shape, -2) #if max_player else 2)
  for action in state.legal_actions():
    if len(tosolve)==0 or (i==0 and action in tosolve):
      values[action] = solve_game(state.child(action),i+1)
  value = values.max() #if max_player else values.min()
  
  best_actions = np.where((values == value) & (act_mask == 1))
  policy = np.zeros_like(act_mask)
  policy[best_actions] = 1/len(best_actions)  # Choose the first for a deterministic policy.
  if not i==0:
    if max_player:
      solved[state_str] = (obs,act_mask, policy, value)
    else:
      solved[state_str] = (obs,act_mask, policy, value)
  if i==1:
    print(i)  
  return -value


if __name__ == "__main__":
    print(123)
    game = main.MNKGame(width=4,height=4,n_in_row=4 )
    state = main.MNKState(game)
#     st = """x..
# o..
# o.x"""
#     state = main.MNKState.emulate_state(game,st)
    for rn in range(1):
      solved = {}
      print(rn)
      solve_game(state,0,[rn])
      a = solved
      #exit()
      # try: 
      #   a_file = open("data444_0.pkl", "wb")
      #   pickle.dump(solved, a_file)
      # except:
      #   pass 

      #rshap =len(solved)+ solved.items()[0][1].shape
      tp = np.dtype([('key', 'U20'), ('obs', 'f8', (3,4,4)),('mask','f8',(16)),('policy','f8',(16)),('value','f8')])
      X = np.zeros(len(a.items()), dtype=tp)
      i = -1
      for k,v in a.items():
        i+=1
        X[i]["key"]=k
        X[i]["obs"] = v[0]
        X[i]["mask"] = v[1]
        X[i]["policy"]=v[2]
        X[i]["value"]=v[3]
        
      
      
      
      
      np.save("np444_"+str(rn), X,True,False)