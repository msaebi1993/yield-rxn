import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Linear


class YieldScoring(nn.Module):
    def __init__(self, hidden_size, binary_size, dmfeats_size, use_domain):
        super(YieldScoring, self).__init__()
        self.use_domain= use_domain

        self.fclocal = Linear(hidden_size, hidden_size, bias=False)
        self.fcglobal = Linear(hidden_size, hidden_size)
        self.fcbinary = Linear(binary_size, hidden_size, bias=False)
        self.fcscore = Linear(hidden_size, 1)
        #self.dcscore = Linear(hidden_size, 1)
        #self.domain= Linear(118, hidden_size)
        #self.finalscore = Linear(2,1)
        
        
    def forward(self, local_features, global_features, binary, sparse_idx,domain_feats):
        l1, l2, l3 ,l4 = binary.shape
        
        global_feats= self.fcglobal(global_features)
        local_feats= self.fclocal(local_features)
        d1, d2, d3 = global_feats.shape
        
        binary_feats = binary.reshape(l1, l2, l3*l4)
        
        padded_binary_feats = F.pad(input=self.fcbinary(binary_feats), pad=(0, 0,  0, d2-l2), mode='constant', value=0)

        #pair_feats = F.relu(self.fclocal(local_pair) + self.fcglobal(global_pair))# + self.fcbinary(binary_feats))
        features= F.relu(global_feats +local_feats +padded_binary_feats)# + self.fcbinary(binary_feats))
        
        l1, l2 = domain_feats.shape        
        #if self.use_domain=='True' or self.use_domain=='only_domain':
            #padded_domain_feats = F.pad(input=domain_feats, pad=(0, 400-l2), mode='constant', value=0)  # domain feature sizes vary based on number of atoms present, so padding it to avoid errors
        
            #domain_features = self.domain( domain_feats).unsqueeze(1)
            
            #if self.use_domain=='only_domain':
                #final_features = domain_features
            #if self.use_domain=='True':
                #final_features=torch.cat((features,domain_features ),dim=1)
       #else:
            #domain_features= torch.zeros(l1, 133)
            #final_features=features
        #final_features=features
        #score = self.fcscore(features)
        #final_score=(torch.mean(torch.abs(score),dim=1))

        if self.use_domain=='True':
            domain_features = self.domain( domain_feats)#.unsqueeze(1)
            score_g = self.fcscore(features)
            score_d= self.dcscore(domain_features)
            #print(domain_features.shape)
            #print(features.shape)
            #print(score_g.shape)
            #print(score_d.shape)
            #final_score=lll
            score_d_avg=torch.mean(torch.abs(score_g),dim=1)
            final_score=(score_d_avg+score_d)/2
            
            #tmp=torch.cat([score_d_avg,score_d],dim=1)
            #final_score=self.finalscore(tmp)
            
            #print(final_score)
            return final_score
        else:
            score_g = self.fcscore(features)
            #final_score=torch.mean(torch.abs(score_g),dim=1)
            final_score=torch.mean(torch.sigmoid(score_g),dim=1)
            return final_score
            #print(final_score.shape)
        #elif self.use_domain=='only_domain':
            #domain_features = self.domain( domain_feats).unsqueeze(1)
            #score_d= self.dcscore(domain_features)
            #final_score= score_d
            #print('blah')
        #else:
            #score_g = self.fcscore(features)
            #final_score = score_g
            #print('crap')
            
        



