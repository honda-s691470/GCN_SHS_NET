import torch.nn as nn
import torch
import torch.nn.functional as F

__all__ = ['GCN_E2_decline_leaky_slope']

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
           

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
class GCN_E2_decline_leaky_slope(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, slope):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, int(in_dim/2))    
        self.gc2 = GraphConvolution(int(in_dim/2), int(in_dim/4))
        self.dropout = dropout
        self.slope = slope
        self.clf = nn.Sequential(
            nn.Linear(int(in_dim/4), int(in_dim/8)),
            nn.LeakyReLU(slope),
            nn.Linear(int(in_dim/8), out_dim))
        self.clf.apply(xavier_init)
        
    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, self.slope)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, self.slope)
        x = self.clf(x)
        return x