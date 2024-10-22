import configparser
import copy

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import kneighbors_graph
import numpy as np
from models.BiaTCGNet.BiaTCGNet import Model
import models
import argparse
import os
import yaml
from data.GenerateDataset import loaddataset
# from tsl.data.utils import WINDOW
import datetime

class Main:
    def __init__(self, args=None):

        torch.multiprocessing.set_sharing_strategy('file_system')
        node_number=7
        self.node_number=node_number
        self.config = Config(self.node_number)
        # Connectivity params
        # parser.add_argument("--adj-threshold", type=float, default=0.1)
        self.args = self.config
        self.criteron=nn.L1Loss().cuda()

        '''if(args.dataset=='Metr'):
            node_number=207
            args.num_nodes=207
            args.enc_in=207
            args.dec_in=207
            args.c_out=207
        elif(args.dataset=='PEMS'):
            node_number=325
            args.num_nodes=325
            args.enc_in = 325
            args.dec_in = 325
            args.c_out = 325
        elif(args.dataset=='ETTh1'):
            node_number=7
            args.num_nodes=7
            args.enc_in = 7
            args.dec_in = 7
            args.c_out = 7
        elif(args.dataset=='Elec'):
            node_number=321
            args.num_nodes=321
            args.enc_in = 321
            args.dec_in = 321
            args.c_out = 321
        elif(args.dataset=='BeijingAir'):
            node_number=36
            args.num_nodes=36
            args.enc_in = 36
            args.dec_in = 36
            args.c_out = 36'''

    def train(self, model, dataset):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        if self.args.seed < 0:
            self.args.seed = np.random.randint(1e9)
        torch.set_num_threads(1)
        #exp_name = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        #exp_name = f"{exp_name}_{args.seed}"
        #logdir = os.path.join('./log_dir', args.dataset_name,
        #                      args.model_name, exp_name)
        # save config for logging
        #os.makedirs(logdir, exist_ok=True)

        train_dataloader, val_dataloader, self.test_dataloader, scaler = loaddataset(self.args.seq_len, self.args.pred_len, self.args.mask_ratio, dataset)
        self.scaler = scaler
        best_loss=9999999.99
        k=0
        for epoch in range(self.args.epochs):
            model.train()
            for i, (x, y, mask, target_mask) in enumerate(train_dataloader):

                x, y, mask,target_mask =x.cuda(), y.cuda(), mask.cuda(), target_mask.cuda()
                x=x*mask
                y=y*target_mask
                x_hat=model(x,mask,k)
                loss = torch.sum(torch.abs(x_hat-y)*target_mask)/torch.sum(target_mask)
                optimizer.zero_grad()  # optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss=self.evaluate(model, val_dataloader,scaler)
            print('epoch, loss:',epoch,loss)
            if(loss<best_loss):
                best_loss=loss
                #best_model = copy.deepcopy(model.state_dict())
                #os.makedirs('./output_BiaTCGNet_'+args.dataset+'_miss'+str(args.mask_ratio)+'_'+args.task,exist_ok=True)
                #torch.save(best_model, './output_BiaTCGNet_'+args.dataset+'_miss'+str(args.mask_ratio)+'_'+args.task+'/best.pth')


    def evaluate(self, model, val_iter,scaler):
        model.eval()
        loss=0.0
        k=0
        with torch.no_grad():
            for i, (x,y,mask,target_mask) in enumerate(val_iter):
                x, y, mask,target_mask = x.cuda(), y.cuda(), mask.cuda(), target_mask.cuda()

                x_hat=model(x,mask,k)

                x_hat = scaler.inverse_transform(x_hat)
                y = scaler.inverse_transform(y)

                losses = torch.sum(torch.abs(x_hat-y)*target_mask)/torch.sum(target_mask)
                loss+=losses


        return loss/len(val_iter)

    def test(self, model):
        model.eval()
        loss=0.0
        all_predictions = []
        k=0
        with torch.no_grad():
            for i, (x,y,mask,target_mask) in enumerate(self.test_dataloader):
                x, y, mask,target_mask = x.cuda(), y.cuda(), mask.cuda(), target_mask.cuda()

                x_hat=model(x,mask,k)

                x_hat = self.scaler.inverse_transform(x_hat)
                y = self.scaler.inverse_transform(y)
                all_predictions.append(y)

                losses = torch.sum(torch.abs(x_hat-y)*target_mask)/torch.sum(target_mask)
                loss+=losses
            print('epoch, loss:', loss)
        return all_predictions

    def run(self, dataset):

        model=Model(True, True, 2, self.node_number, self.args.kernel_set,
                'cuda:0', predefined_A=None,
                dropout=0.3, subgraph_size=5,
                node_dim=3,
                dilation_exponential=1,
                conv_channels=8, residual_channels=8,
                skip_channels=16, end_channels= 32,
                seq_length=self.args.seq_len, in_dim=1,out_len=self.args.pred_len, out_dim=1,
                layers=2, propalpha=0.05, tanhalpha=3, layer_norm_affline=True) #2 4 6
        if torch.cpu.is_available():
            model = model.cpu()
        else:
            model = model.cuda()

        self.train(model, dataset)
        prediction = self.test(model)
        print(prediction)
class Config:
    def __init__(self, node_number):
        # Training parameters
        self.epochs = 100
        self.batch_size = 64
        self.task = 'prediction'
        self.adj_threshold = 0.1
        self.val_ratio = 0.2
        self.test_ratio = 0.2
        self.column_wise = False
        self.seed = -1
        self.precision = 32
        self.model_name = 'spin'
        self.dataset_name = 'air36'
        self.fc_dropout = 0.2
        self.head_dropout = 0
        self.individual = 0  # True 1 False 0
        self.patch_len = 8
        self.padding_patch = 'end'  # None: None; end: padding on the end
        self.revin = 0  # RevIN; True 1 False 0
        self.affine = 0  # RevIN-affine; True 1 False 0
        self.subtract_last = 0  # 0: subtract mean; 1: subtract last
        self.decomposition = 0  # decomposition; True 1 False 0
        self.kernel_size = 25
        self.kernel_set = [2, 3, 6, 7]

        # Transformer config
        self.enc_in = node_number  # Replace with actual node number
        self.dec_in = node_number  # Replace with actual node number
        self.c_out = node_number  # Replace with actual node number
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 2
        self.d_ff = 2048
        self.moving_avg = [24]
        self.factor = 1
        self.dropout = 0.05
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.freq = 'h'
        self.num_nodes = node_number  # Replace with actual node number
        self.version = 'Fourier'
        self.mode_select = 'random'
        self.modes = 64
        self.L = 3
        self.base = 'legendre'
        self.cross_activation = 'tanh'

        # AGCRN
        self.input_dim = 1
        self.output_dim = 1
        self.embed_dim = 512
        self.rnn_units = 64
        self.num_layers = 2
        self.cheb_k = 2
        self.default_graph = True

        # GTS
        self.temperature = 0.5
        self.config_filename = ''
        self.config = 'imputation/spin.yaml'
        self.output_attention = False
        self.val_len = 0.2
        self.test_len = 0.2
        self.mask_ratio = 0.1

        # Training params
        self.lr = 0.001
        self.patience = 40
        self.l2_reg = 0.0
        self.batch_inference = 32
        self.split_batch_in = 1
        self.grad_clip_val = 5.0
        self.loss_fn = 'l1_loss'
        self.lr_scheduler = None
        self.seq_len = 24
        self.label_len = 12
        self.pred_len = 24
        self.horizon = 24
        self.delay = 0
        self.stride = 1
        self.window_lag = 1
        self.horizon_lag = 1
