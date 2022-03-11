import numpy as np
a = 0
_NUM_PLAYERS = 2
# _NUM_ROWS = 3
# _NUM_COLS = 3
# _NUM_CELLS = _NUM_ROWS * _NUM_COLS

short_name = "python_mnk"
long_name = "Python MNK"
dynamics = "SEQUENTIAL"
chance_mode = "DETERMINISTIC",
# information=pyspiel.GameType.Information.PERFECT_INFORMATION,
# utility=pyspiel.GameType.Utility.ZERO_SUM,
# reward_model=pyspiel.GameType.RewardModel.TERMINAL,
# max_num_players=_NUM_PLAYERS,
# min_num_players=_NUM_PLAYERS,
# provides_information_state_string=True,
# provides_information_state_tensor=False,
# provides_observation_string=True,
# provides_observation_tensor=True,
# parameter_specification={})

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

class MNKGame:
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        # self.states = {}
        # # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        # self.players = [1, 2]  # player1 and player2

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return MNKState(self)
    
    def max_game_length(self):
        return self.height*self.width

    #def init_board(self, start_player=0):
    #     if self.width < self.n_in_row or self.height < self.n_in_row:
    #         raise Exception('board width and height can not be '
    #                         'less than {}'.format(self.n_in_row))
    #     self.current_player = self.players[start_player]  # start player
    #     # keep available moves in a list
    #     self.availables = list(range(self.width * self.height))
    #     self.states = {}
    #     self.last_move = -1

class MNKState():
    
    def _coord(self, move):
        """Returns (row, col) from an action id."""
        return (move // self.game.width, move % self.game.width)
    
    def _index(self,coord):
        return self.width*coord[0] + coord[1];

    def _board_to_string(self, board):
        """Returns a string representation of the board."""
        return "\n".join("".join(row) for row in board)
        
    
    def _line_exists(self,board):
        kNumCells = self.number_of_cells
        kNumRows = self.game.height
        kNumCols = self.game.width
        kWin = self.game.n_in_row
        mainDiagWin=np.zeros(self.number_of_cells)
        secondDiagWin=np.zeros(self.number_of_cells)
        rowWin = np.zeros(self.number_of_cells)
        colWin = np.zeros(self.number_of_cells)
        found = False
        c = 'x'
        if self._cur_player==1:
            c = 'o'
        for  i in range(0,kNumCells):
          move = self._coord(i)
          if board[move] == c:
            curCol = i % kNumRows
            #//rows
            if i % kNumCols > 0 and i > 0:
                rowWin[i] += rowWin[i - 1] + 1;
            else:
                #//newlines
                rowWin[i] = 1;
          

            if rowWin[i] >= kWin:
           # //std::cout << "row win\n"<< i << std::endl;

                found = True
                break
          
          #//cols
            if i >= kNumCols:
                colWin[i] += colWin[i - kNumCols] + 1
            else:
                colWin[i] = 1
          
            if colWin[i] >= kWin:
            #//std::cout << "Col win"<< i << std::endl;

                found = True
                break
          

          #//main diag
            if i > kNumCols:
                mainDiagWin[i] += mainDiagWin[i - kNumCols - 1] + 1
            
            else:
                mainDiagWin[i] = 1
            if mainDiagWin[i] >= kWin:
                #//std::cout << "main win" << i << std::endl;
                found = True
                break
            
            #//secondary diag
            if i > kNumCols and curCol != (kNumCols - 1):
                
                    secondDiagWin[i] += secondDiagWin[i - kNumCols + 1] + 1
                
            else:
                
                    secondDiagWin[i] = 1
                
            if secondDiagWin[i] >= kWin:
            
                #//std::cout << "second win" << i << std::endl;

                found = True
                break;
            
        
        else:
            rowWin[i] = 0;
            colWin[i] = 0;
            mainDiagWin[i] = 0;
            secondDiagWin[i] = 0;
        
      

        return found
    


    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""

        self._cur_player = 0
        self.game = game
        self.width = game.width
        self.height = game.height
        self.n_in_row = game.n_in_row
        self._player0_score = 0.0
        self._is_terminal = False
        self.number_of_cells = game.height * game.width
        self.board = np.full((game.height, game.width), ".")
        self.additionals = [{},{}]
        self.additionals[0]["win_hor"] = np.full((game.height, game.width), 0)
        self.additionals[0]["win_ver"] = np.full((game.height, game.width), 0)
        self.additionals[0]["win_main"] = np.full((game.height, game.width), 0)
        self.additionals[0]["win_sec"] = np.full((game.height, game.width), 0)

        self.additionals[1]["win_hor"] = np.full((game.height, game.width), 0)
        self.additionals[1]["win_ver"] = np.full((game.height, game.width), 0)
        self.additionals[1]["win_main"] = np.full((game.height, game.width), 0)
        self.additionals[1]["win_sec"] = np.full((game.height, game.width), 0)

        self.legal =list(range(0, self.number_of_cells ))

    def new_initial_state(self):
        self._cur_player = 0
        game = self.game
        self.game = game
        self.width = game.width
        self.height = game.height
        self.n_in_row = game.n_in_row
        self._player0_score = 0.0
        self._is_terminal = False
        self.number_of_cells = game.height * game.width
        self.board = np.full((game.height, game.width), ".")
        self.additionals = [{},{}]
        self.additionals[0]["win_hor"] = np.full((game.height, game.width), 0)
        self.additionals[0]["win_ver"] = np.full((game.height, game.width), 0)
        self.additionals[0]["win_main"] = np.full((game.height, game.width), 0)
        self.additionals[0]["win_sec"] = np.full((game.height, game.width), 0)

        self.additionals[1]["win_hor"] = np.full((game.height, game.width), 0)
        self.additionals[1]["win_ver"] = np.full((game.height, game.width), 0)
        self.additionals[1]["win_main"] = np.full((game.height, game.width), 0)
        self.additionals[1]["win_sec"] = np.full((game.height, game.width), 0)

        self.legal =list(range(0, self.number_of_cells ))


    def num_distinct_actions(self):
        return self.number_of_cells

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        return -1 if self._is_terminal else self._cur_player

    def _legal_actions(self,player):
        """Returns a list of legal actions, sorted in ascending order."""
        return self.legal
        #return [a for a in range(self.number_of_cells) if self.board[self._coord(a)] == "."]
    
    def legal_actions(self, player = None):
        return self.legal

    def legal_actions_mask(self, player = None):
        a = np.zeros(self.width*self.height);
        for action in self.legal_actions():
            a[action] = 1;
        return a
        #return self._legal_actions(player)

    def observation_tensor(self):
        res = np.zeros([3,self.height,self.width])
        ch1 = 'x'
        ch2 = 'o'
        # if self.current_player() == 1:
        #     ch1 = 'o'
        #     ch2 = 'x'

        r1 = np.where(self.board==ch1,1,0)
        r2 = np.where(self.board==ch2,1,0)
        r3 = r1 + (-1*r2)

        res[0] = r1
        res[1] = r2
        res[2] = r3
        return res

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        self.legal.remove(action)
        coords = self._coord(action)
        self.board[coords] = "x" if self._cur_player == 0 else "o"
        found_victor = False
        additionals = self.additionals[self._cur_player]
        
        a = 0
        b = 0
        if coords[1]>0:
            a = additionals["win_hor"][coords[0],coords[1]-1]
        if coords[1]<self.width-1:
            b = additionals["win_hor"][coords[0],coords[1]+1]
        sum_val = a + b + 1
        additionals["win_hor"][coords] = sum_val;
        #goleft
        ncoords = coords
        ncoords=(ncoords[0],ncoords[1]-1)
        while ncoords[1]>=0 and additionals["win_hor"][ncoords]!=0:
            additionals["win_hor"][ncoords] = sum_val
            ncoords=(ncoords[0],ncoords[1]-1)
        #goright
        ncoords = coords
        ncoords=(ncoords[0],ncoords[1]+1)
        while ncoords[1]<self.width and additionals["win_hor"][ncoords]!=0:
            additionals["win_hor"][ncoords] = sum_val
            ncoords=(ncoords[0],ncoords[1]+1)
            


        if sum_val >= self.n_in_row:
            found_victor = True

        a = 0
        b = 0
        if coords[0]>0:
            a = additionals["win_ver"][coords[0]-1,coords[1]]
        if coords[0]<self.height-1:
            b = additionals["win_ver"][coords[0]+1,coords[1]]
        sum_val = a + b + 1
        additionals["win_ver"][coords] = sum_val
         #goup
        ncoords = coords
        ncoords=(ncoords[0]-1,ncoords[1])
        while ncoords[0]>=0 and additionals["win_ver"][ncoords]!=0:
            additionals["win_ver"][ncoords] = sum_val
            ncoords=(ncoords[0]-1,ncoords[1])
        #godown
        ncoords = coords
        ncoords=(ncoords[0]+1,ncoords[1])
        while ncoords[0]<self.height and additionals["win_ver"][ncoords]!=0:
            additionals["win_ver"][ncoords] = sum_val
            ncoords=(ncoords[0]+1,ncoords[1])
        if sum_val >= self.n_in_row:
            found_victor = True

        a = 0
        b = 0
        if coords[1]>0 and coords[0]>0:
            a = additionals["win_main"][coords[0]-1,coords[1]-1]
        if coords[1]<self.width-1 and coords[0]<self.height-1:
            b = additionals["win_main"][coords[0]+1,coords[1]+1]
        sum_val = a + b + 1
        additionals["win_main"][coords] = sum_val
        
        #goupleft
        ncoords = coords
        ncoords=(ncoords[0]-1,ncoords[1]-1)
        while ncoords[0]>=0 and ncoords[1]>=0 and additionals["win_main"][ncoords]!=0:
            additionals["win_main"][ncoords] = sum_val
            ncoords=(ncoords[0]-1,ncoords[1]-1)
        #godownright
        ncoords = coords
        ncoords=(ncoords[0]+1,ncoords[1]+1)
        while ncoords[0]<self.height and ncoords[1]<self.width and additionals["win_main"][ncoords]!=0:
            additionals["win_main"][ncoords] = sum_val
            ncoords=(ncoords[0]+1,ncoords[1]+1)
        
        if sum_val >= self.n_in_row:
            found_victor = True

        a = 0
        b = 0
        if coords[1]>0 and  coords[0]<self.width-1:
            a = additionals["win_sec"][coords[0]+1,coords[1]-1]
        if coords[1]<self.height-1 and coords[0]>0:
            b = additionals["win_sec"][coords[0]-1,coords[1]+1]
        sum_val = a + b + 1
        additionals["win_sec"][coords] = sum_val;
        #goupright
        ncoords = coords
        ncoords=(ncoords[0]-1,ncoords[1]+1)
        while ncoords[0]>=0 and ncoords[1]<self.width and additionals["win_sec"][ncoords]!=0:
            additionals["win_sec"][ncoords] = sum_val
            ncoords=(ncoords[0]-1,ncoords[1]+1)
        #godownright
        ncoords = coords
        ncoords=(ncoords[0]+1,ncoords[1]-1)
        while ncoords[0]<self.height and ncoords[1]>=0 and additionals["win_sec"][ncoords]!=0:
            additionals["win_sec"][ncoords] = sum_val
            ncoords=(ncoords[0]+1,ncoords[1]-1)
        if sum_val >= self.n_in_row:
            found_victor = True

        if found_victor:
            self._is_terminal = True
            self._player0_score = 1.0 if self._cur_player == 0 else -1.0
        elif all(self.board.ravel() != "."):
            self._is_terminal = True
        else:
         self._cur_player = 1 - self._cur_player
    
    def apply_action(self,action):
        return self._apply_action(action)

    def _action_to_string(self, player, action):
        """Action -> string."""
        row, col = self._coord(action)
        return "{}({},{})".format("x" if player == 0 else "o", row, col)

    def action_to_string(self, player, action):
       return self._action_to_string(player,action)

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal
    
    def clone(self):
        res = MNKState(self.game)
        res._cur_player = self._cur_player
       
        res._player0_score =self._player0_score
        res._is_terminal = self._is_terminal
        res.legal = self.legal.copy()
        res.board = np.copy(self.board)
        res.additionals[0]["win_hor"] = np.copy(self.additionals[0]["win_hor"])
        res.additionals[0]["win_ver"] = np.copy(self.additionals[0]["win_ver"])
        res.additionals[0]["win_main"]= np.copy(self.additionals[0]["win_main"])
        res.additionals[0]["win_sec"] = np.copy(self.additionals[0]["win_sec"])

        res.additionals[1]["win_hor"] = np.copy(self.additionals[1]["win_hor"])
        res.additionals[1]["win_ver"] = np.copy(self.additionals[1]["win_ver"])
        res.additionals[1]["win_main"]= np.copy(self.additionals[1]["win_main"])
        res.additionals[1]["win_sec"] = np.copy(self.additionals[1]["win_sec"])

        return res

    def child(self,action):
        new_state= self.clone()
        new_state.apply_action(action)
        return new_state


    
    @staticmethod
    def emulate_state(game,state_string):
        #might NOT work for terminal states!!!
        res = MNKState(game)

        a = state_string.replace('\n','')
        i = -1
        x_ind = find(a,'x')
        o_ind = find(a,'o')
        for ind in x_ind:
            i+=1
            res.apply_action(ind)
            if i<len(o_ind):
                res.apply_action(o_ind[i])
        return res

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return [self._player0_score, -self._player0_score]

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return self._board_to_string(self.board)


if __name__ == '__main__':
    print(123)
    game = MNKGame(width=3,height=3,n_in_row=3 )
    state = MNKState(game)
    while not state.is_terminal():         
        print(state)

        print(MNKState.emulate_state(game,str(state)))

        print(state.is_terminal())
        print(state.legal_actions_mask())
        num1 = int(input())
        state._apply_action(num1)
        print(state.observation_tensor())
        print(state.legal_actions())
    print(state._player0_score)