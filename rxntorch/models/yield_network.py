import logging
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
from .wln import WLNet
from .attention import Attention
from .yield_scoring import YieldScoring
from sklearn.metrics import roc_auc_score,r2_score

class YieldNet(nn.Module):
    def __init__(self, depth, dropout, afeats_size, bfeats_size, hidden_size, binary_size,dmfeats_size, use_domain):
        super(YieldNet, self).__init__()
        self.hidden_size = hidden_size
        self.wln = WLNet(depth, dropout, afeats_size, bfeats_size, hidden_size)
        
        self.attention = Attention(hidden_size, binary_size)
        self.yield_scoring= YieldScoring(hidden_size, binary_size, dmfeats_size, use_domain)
        self.dropout = nn.Dropout(dropout)

    def forward(self, fatoms, fbonds, atom_nb, bond_nb, num_nbs, n_atoms, binary_feats, mask_neis, mask_atoms, sparse_idx,domain_feats):
        #def forward(self, fatoms, fbonds, atom_nb, bond_nb, num_nbs, n_atoms, binary_feats, mask_neis, mask_atoms, sparse_idx):
        local_features = self.wln(fatoms, fbonds, atom_nb, bond_nb, num_nbs, n_atoms, mask_neis, mask_atoms)
        #local_pair, global_pair = self.attention(local_features, binary_feats, sparse_idx)
        d_local_features = self.dropout(local_features)
        global_features = self.attention(local_features, binary_feats, sparse_idx)
        d_global_features= self.dropout(global_features)
        #yield_scores = self.yield_scoring(local_features, global_features, binary_feats, sparse_idx)
        yield_scores = self.yield_scoring(d_local_features, d_global_features, binary_feats, sparse_idx, domain_feats)
        #yield_scores = self.yield_scoring(local_features, 0, binary_feats, sparse_idx)
        

        return yield_scores


class YieldTrainer(nn.Module):
    def __init__(self, rxn_net, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0, with_cuda=True,
                 cuda_devices=None, log_freq=10, grad_clip=None, pos_weight=1.0, lr_decay=0.9,
                 lr_steps=10000):
        super(YieldTrainer, self).__init__()
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        self.model = rxn_net
        if cuda_condition and (torch.cuda.device_count() > 1):
            logging.info("Using {} GPUS".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        self.model.to(self.device)
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_steps = lr_steps
        self.grad_clip = grad_clip
        self.log_freq = log_freq
        self.pos_weight = pos_weight
        self.total_iters = 0
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr, betas=betas, weight_decay=weight_decay)
        #self.lr_scheduler = opt.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=1-self.lr_decay)
        self.lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True,factor=self.lr_decay,min_lr=1e-6)
        
    def train_epoch(self, epoch, data_loader):
        logging.info("{:-^80}".format("Training"))
        self.model.train()
        #r2 ,loss,w1,w2= self.iterate(epoch, data_loader, train=True,valid=False)
        r2 ,loss= self.iterate(epoch, data_loader, train=True,valid=False)
        return r2,loss#,w1,w2

    def test_epoch(self, epoch, data_loader):
        #logging.info("{:-^80}".format("Testing"))
        with torch.no_grad():
            self.model.eval()
            r2 ,loss= self.iterate(epoch, data_loader, train=False,valid=False)
        return r2,loss

    def valid_epoch(self, epoch, data_loader):
        logging.info("{:-^80}".format("Validating"))
        with torch.no_grad():
            self.model.eval()
        r2 ,loss= self.iterate(epoch, data_loader, train=False,valid=True)
        return r2,loss


    def iterate(self, epoch, data_loader, train=True,valid=False):
        avg_loss = test_loss = 0.0
        sum_gnorm = 0.0
        tmp_r2=0
        iters = len(data_loader)
        n_samples = len(data_loader.dataset)
        r2=0
        correct_yields=np.array([[0]])
        pred_yields=np.array([[0]])
        cum_r2=0
        for i, data in enumerate(data_loader):
            data = {key: value.to(self.device) for key, value in data.items()}

            # Create some masking logic for padding
            mask_neis = torch.unsqueeze(
                data['n_bonds'].unsqueeze(-1) > torch.arange(0, 15, dtype=torch.int32, device=self.device).view(1, 1, -1), -1)
            max_n_atoms = data['n_atoms'].max()
            mask_atoms = torch.unsqueeze(
                data['n_atoms'].unsqueeze(-1) > torch.arange(0, max_n_atoms, dtype=torch.int32, device=self.device).view(1, -1),
                -1)
            
            yield_scores = self.model.forward(data['atom_feats'], data['bond_feats'],data['atom_graph'], 
                                                data['bond_graph'], data['n_bonds'],data['n_atoms'], data['binary_feats'], 
                                                mask_neis, mask_atoms,data['sparse_idx'],data['domain_feats'])
            

            criteria=nn.MSELoss()
            loss= criteria(yield_scores, data['yield_label'])
            loss = torch.mean(loss)
            avg_loss += loss.item()
            test_loss += loss.item()
            #if True:
            aa=data['yield_label'].cpu().detach().numpy()
            bb=yield_scores.cpu().detach().numpy()
            correct_yields=np.append(correct_yields,aa,0)
            pred_yields=np.append(pred_yields,bb,0)
                
            if aa.shape !=bb.shape or correct_yields.shape != pred_yields.shape:
                raise ValueError("Found input variables with inconsistent numbers of elements")
                
            tmp_r2=r2_score(correct_yields,pred_yields)
            cum_r2 += r2_score(aa,bb)
                
            tmp_r2=0 if math.isnan(tmp_r2) else tmp_r2
            cum_r2=0 if math.isnan(cum_r2) else cum_r2
            learning_rate=np.mean([ group['lr'] for group in self.optimizer.param_groups ])
            if train:
                self.total_iters += 1
                self.optimizer.zero_grad()
                loss.backward()
                
                param_norm = torch.sqrt(sum([torch.sum(param ** 2) for param in self.model.parameters()])).item()
                grad_norm = torch.sqrt(sum([torch.sum(param.grad ** 2) for param in self.model.parameters()])).item()
                sum_gnorm += grad_norm
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                                
            if (i+1) % self.log_freq == 0:
                if train:
                    post_fix = {
                            "epoch": epoch,
                            "iter": (i+1),
                            "iters": iters,
                            "avg_loss": avg_loss,
                            "r2": tmp_r2,
                            #"cum_r2":cum_r2/i,
                            "pnorm": param_norm,
                            "gnorm": sum_gnorm,
                            "learning_rate":learning_rate
                    }
                    logging.info(("Epoch: {epoch:2d}  Iter: {iter:5d}  Loss: {avg_loss:7.5f}  R2: {r2:6.2%}  LR:{learning_rate:8.8f} " 
                            "Param norm: {pnorm:8.4f}  Grad norm: {gnorm:8.4f}").format(
                            **post_fix))
                    #learned_weights=self.model.module.yield_scoring.finalscore.weight.data
                    #w1,w2=learned_weights[0][0].item(),learned_weights[0][1].item()
                    #logging.info(learned_weights)
                    
                if not train:
                    post_fix = {
                            "epoch": epoch,
                            "iter": (i+1),
                            "iters": iters,
                            "avg_loss": avg_loss,
                            "r2": tmp_r2,
                            #"cum_r2":cum_r2/i
                            "learning_rate": learning_rate
                        }
                    logging.info(("Epoch: {epoch:2d}  Iter: {iter:5d}  Loss: {avg_loss:7.5f}  R2: {r2:6.2%}  LR:{learning_rate:8.8f} ").format(
                            **post_fix))
                    
                avg_loss = 0.0
                sum_gnorm = 0.0
            """ 
            if self.total_iters % self.lr_steps == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.lr_decay
                logging.info("Learning rate changed to {:f}".format(self.optimizer.param_groups[0]['lr']))
            """   
        if valid:
            self.lr_scheduler.step(avg_loss)
            logging.info("Epoch: {:2d}  valid Loss: {:f}  Valid R2: {:6.2%}  LR:{:8.8f}  ".format(epoch,test_loss, tmp_r2,learning_rate))
            return tmp_r2,avg_loss
        if train:
            #learned_weights=self.model.module.yield_scoring.finalscore.weight.data
            #w1,w2=learned_weights[0][0].item(),learned_weights[0][0].item()
            return tmp_r2,avg_loss#,w1,w2
        if not train and not valid:
            logging.info("Epoch: {:2d}  Loss: {:f}  R2: {:6.2%}  LR:{:8.8f}  ".format(epoch,test_loss, tmp_r2,learning_rate))
            return tmp_r2, test_loss

        

        
            
    def save(self, epoch, filename, path):
        #filename = filename + ".ep%d" % epoch
        #output_path=path + filename +'/yield'+'_'+str(epoch)+'.model'
        output_path=path + filename +'/yield.model'
        #output = os.path.join(path, filename)
        torch.save(self.model.module.cpu(), output_path)
        self.model.to(self.device)
        logging.info("Model saved to {}:".format(output_path))

