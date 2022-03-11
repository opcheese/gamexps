import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import main

import collections
import functools
import os

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def double_conv(in_c,out_c):
  conv = nn.Sequential(
      nn.Conv2d(in_c,out_c,kernel_size=3),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_c,out_c,kernel_size=3),
      nn.ReLU(inplace=True)
  )
  return conv

def crop_img(tensor, target_tensor):
  target_size = target_tensor.size()[2]
  tensor_size = tensor.size()[2]
  delta = tensor_size - target_size
  delta = delta//2
  return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]
                                    



class SimpleNet(nn.Module):
    def __init__(self, out_sz=(20,20),ch=3):
        super().__init__()

        wer=123
        self.conv1 = nn.Conv2d(ch, 8, 3, padding = 1)
        #self.bn1 =  nn.BatchNorm2d(m_ch)
        #self.drop1 = nn.Dropout2d(0.2)        
        self.relu1  = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 3,  padding = 1)
        #self.bn2 =  nn.BatchNorm2d(m_ch)
        self.drop2 = nn.Dropout2d(0.1)
        self.relu2  = nn.ReLU()
        
        self.super_head_policy = nn.Linear(16*out_sz[0]*out_sz[1],out_sz[0]*out_sz[1])
        
        self.max_policy = nn.Softmax()
        self.super_head_value = nn.Linear(16*out_sz[0]*out_sz[1],1)
        self.tahn_value = nn.Tanh()

    def forward(self, x):
        x =  self.relu1(self.conv1(x))
        x =  self.relu2(self.drop2(self.conv2(x)))
        x =  torch.flatten(x, 1)

        out1 = self.super_head_policy(x)
        out1 = self.max_policy(out1)

        out2 = self.super_head_value(x)
        out2 = self.tahn_value(out2)

        return out1,out2


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True, path = "simplenet"):
        self.use_gpu = use_gpu
        self.path = path
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = SimpleNet((board_width, board_height)).cuda()
        else:
            self.policy_value_net = SimpleNet((board_width, board_height))
       
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = log_act_probs.data.cpu().numpy()
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = log_act_probs.data.numpy()
            return act_probs, value.data.numpy()

    def policy_value_fn(self, state):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        board = state.board
        legal_positions = state.legal_actions
        current_state = np.ascontiguousarray(board.reshape(
                -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = log_act_probs.data.cpu().numpy().flatten()
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = log_act_probs.data.numpy().flatten()
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def inference(self, obs, mask):
        self.policy_value_net.eval()
        policy,value =  self.policy_value_net(obs)
        zs = torch.zeros_like(mask)
        policy = torch.where(mask==1.0,policy,zs)
        return policy,value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        self.loss = loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return {"loss": loss.item(),"entripy":entropy.item()}
        #for pytorch version >= 0.5 please use the following line instead.
        #return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)

    def save_checkpoint(self, number):
        """ save model params to file """
        path = self.path+"_" + str(number) +".fmodel"
        net_params = self.get_policy_param()  # get model params
        opt_params = self.optimizer.state_dict(),


        torch.save({
            'model_state_dict': net_params,
            'optimizer_state_dict': opt_params
            }, path)
            
    def load_checkpoint(self,path):
        if self.use_gpu:
            model = SimpleNet(16*self.board_width*self.board_height ,(self.board_width, self.board_height)).cuda()
        else:
            model = SimpleNet(16*self.board_width*self.board_height ,(self.board_width, self.board_height))
       
        optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #loss = checkpoint['loss']


class TrainInput(collections.namedtuple(
    "TrainInput", "observation legals_mask policy value")):
  """Inputs for training the Model."""

  @staticmethod
  def stack(train_inputs):
    observation, legals_mask, policy, value = zip(*train_inputs)
    return TrainInput(
        np.array(observation, dtype=np.float32),
        np.array(legals_mask, dtype=np.bool),
        np.array(policy),
        np.expand_dims(value, 1))


class Losses(collections.namedtuple("Losses", "policy value l2")):
  """Losses from a training step."""

  @property
  def total(self):
    return self.policy + self.value + self.l2

  def __str__(self):
    return ("Losses(total: {:.3f}, policy: {:.3f}, value: {:.3f}, "
            "l2: {:.3f})").format(self.total, self.policy, self.value, self.l2)

  def __add__(self, other):
    return Losses(self.policy + other.policy,
                  self.value + other.value,
                  self.l2 + other.l2)

  def __truediv__(self, n):
    return Losses(self.policy / n, self.value / n, self.l2 / n)

if __name__ == '__main__':
  print(torch.cuda.is_available())
  pvn = PolicyValueNet(3,3)
  print(pvn.policy_value_net)
  ##res = pvn.policy_value_net()

  game = main.MNKGame(width=3,height=3,n_in_row=3 )
  state = main.MNKState(game)
  st = """xox
xoo
.x."""
  state = main.MNKState.emulate_state(game,st)
  tens = state.observation_tensor()
  tens = np.expand_dims(tens, axis=0)
  t = torch.from_numpy(tens)
  t = t.to('cuda')
  t = t.type(torch.cuda.FloatTensor)

  msk = state.legal_actions_mask()
  m = torch.from_numpy(msk)
  m = m.to('cuda')
  m = m.type(torch.cuda.FloatTensor)
  a = pvn.inference(t,m )
  print(a)

