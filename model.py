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
                                    
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, ker = 3):
        super().__init__()
        
        m_ch = out_ch
        if out_ch>in_ch:
          m_ch = out_ch
        self.chs = (in_ch,out_ch,m_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, ker, padding = 1)
        #self.bn1 =  nn.BatchNorm2d(m_ch)
        #self.drop1 = nn.Dropout2d(0.2)        
        self.relu1  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, ker,  padding = 1)
        #self.bn2 =  nn.BatchNorm2d(m_ch)
        self.drop2 = nn.Dropout2d(0.1)        
        
        self.relu2  = nn.ReLU()
        #print(self.conv1)
    
    def forward(self, x):
        #a = self.conv2(self.relu(self.conv1(x)))
        a = self.conv1(x)
       
        #a = self.bn1(a)
        a = self.relu1(a)
        #a = self.drop1(a)
        a = self.conv2(a)
        #a = self.bn2(a)
        a = self.drop2(a)
        #a = self.relu2(a)  
        #print(a.shape)
        return a


class Encoder(nn.Module):
    def __init__(self, chs=(3,32,64,128), ks = [3,3,3,3], pool = 1):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1],ks[i]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(pool)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(128, 64,32), ks = [3,3,3], pool = 1):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], pool, pool) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1],ks[i]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(3,32,64,128), dec_chs=(128, 64, 32), num_class=3, retain_dim=False, out_sz=(8,8), pool = 2):
        super().__init__()
        self.encoder     = Encoder(enc_chs, pool=pool)
        self.decoder     = Decoder(dec_chs, pool = pool)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)

        self.super_head_policy = nn.Linear(out_sz[0]*out_sz[1]*num_class,out_sz[0]*out_sz[1])
        
        self.max_policy = nn.Softmax()
        self.super_head_value = nn.Linear(out_sz[0]*out_sz[1]*num_class,1)
        self.tahn_value = nn.Tanh()
        
        self.retain_dim  = retain_dim

    def forward(self, x,mask):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        out = torch.flatten(out, 1)
        #print(out.shape)
        out1 = self.super_head_policy(out)
        #print(out1.shape)
        #print(out1.shape)
        #print(mask.shape)

        out1 = torch.mul(out1,mask)
        out1 = self.max_policy(out1)

        out2 = self.super_head_value(out)
        out2 = self.tahn_value(out2)
        
 
        # if self.retain_dim:
        #     out = F.interpolate(out, out_sz)
        return out1,out2

class UnetPolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, fmodel_file=None, use_gpu=True, path = "unetsimplenet"):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        pool=2
        if board_height<8 or board_width<9:
          pool = 1
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = UNet(out_sz=(board_width,board_height),pool=pool).cuda()
        else:
            self.policy_value_net = UNet(out_sz=(board_width,board_height),pool=pool)
       
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        self.path = path

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)
        if fmodel_file:
            checkpoint = torch.load(fmodel_file)
            self.policy_value_net.load_state_dict(checkpoint['model_state_dict'])
            #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        

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
      policy,value =  self.policy_value_net(obs,mask)
    #   zs = torch.zeros_like(mask)
    #   policy = torch.where(mask==1.0,policy,zs)
      return policy,value

    def train_step(self, state_batch,mask_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch,mask_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
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
        
        return path
            
    def load_checkpoint(self,path):
        pool=2
        
        if self.board_height<8 or self.board_width<9:
          pool = 1
        if self.use_gpu:
            self.policy_value_net = UNet(out_sz=(self.board_width,self.board_height),pool=pool).cuda()
        else:
            self.policy_value_net = UNet(out_sz=(self.board_width,self.board_height),pool=pool)
       
        optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        checkpoint = torch.load(path)
        self.policy_value_net.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
  pvn = UnetPolicyValueNet(3,3)
  print(pvn.policy_value_net)
  ##res = pvn.policy_value_net()

  game = main.MNKGame(width=3,height=3,n_in_row=3 )
  state = main.MNKState(game)
  tens = state.observation_tensor()
  tens = np.expand_dims(tens, axis=0)
  t = torch.from_numpy(tens)
  t = t.to('cuda')
  t = t.type(torch.cuda.FloatTensor)

  mask = state.legal_actions_mask()
  mask = np.expand_dims(mask, axis=0)
  tm = torch.from_numpy(mask)
  tm = tm.to('cuda')
  tm = tm.type(torch.cuda.FloatTensor)

  a = pvn.policy_value_net(t,tm)
  print(a)