"""GAMENet: Graph-Augmented Memory Network for medication recommendation.

Architecture must exactly match saved checkpoints from medrec_pipeline training.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias   = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, adj):
        out = torch.mm(x, self.weight)
        out = torch.spmm(adj, out) if adj.is_sparse else torch.mm(adj, out)
        return out + self.bias if self.bias is not None else out


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device):
        super().__init__()
        self.device = device
        adj = self._normalize(adj + np.eye(adj.shape[0]))
        self.adj = torch.FloatTensor(adj).to(device)
        self.x   = torch.eye(voc_size).to(device)
        self.gcn1    = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(0.3)
        self.gcn2    = GraphConvolution(emb_dim, emb_dim)

    def _normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv  = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        return np.diagflat(r_inv).dot(mx)

    def forward(self):
        h = F.relu(self.gcn1(self.x, self.adj))
        h = self.dropout(h)
        return self.gcn2(h, self.adj)


class GAMENet(nn.Module):
    def __init__(self, voc_size, ehr_adj, ddi_adj, emb_dim=64,
                 device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.voc_size = voc_size
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        self.embeddings = nn.ModuleList([
            nn.Embedding(voc_size[i], emb_dim) for i in range(2)])
        self.dropout    = nn.Dropout(0.5)
        self.encoders   = nn.ModuleList([
            nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(2)])
        self.query   = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim * 4, emb_dim))
        self.ehr_gcn = GCN(voc_size[2], emb_dim, ehr_adj, device)
        self.ddi_gcn = GCN(voc_size[2], emb_dim, ddi_adj, device)
        self.inter   = nn.Parameter(torch.FloatTensor(1))
        self.output  = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, voc_size[2]),
        )
        for emb in self.embeddings:
            emb.weight.data.uniform_(-0.1, 0.1)
        self.inter.data.uniform_(-0.1, 0.1)

    def forward(self, input_seq):
        i1_seq, i2_seq = [], []
        for adm in input_seq:
            i1 = self.dropout(self.embeddings[0](
                torch.LongTensor(adm[0]).unsqueeze(0).to(self.device))
            ).mean(1).unsqueeze(0)
            i2 = self.dropout(self.embeddings[1](
                torch.LongTensor(adm[1]).unsqueeze(0).to(self.device))
            ).mean(1).unsqueeze(0)
            i1_seq.append(i1); i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, 1); i2_seq = torch.cat(i2_seq, 1)
        o1, _ = self.encoders[0](i1_seq)
        o2, _ = self.encoders[1](i2_seq)
        repr_   = torch.cat([o1, o2], -1).squeeze(0)
        queries = self.query(repr_)
        query   = queries[-1:]

        drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter

        k1    = F.softmax(torch.mm(query, drug_memory.t()), -1)
        fact1 = k1.mm(drug_memory)

        if len(input_seq) > 1:
            hk  = queries[:-1]
            hv  = np.zeros((len(input_seq) - 1, self.voc_size[2]))
            for i, adm in enumerate(input_seq[:-1]):
                hv[i, adm[2]] = 1
            hv_t  = torch.FloatTensor(hv).to(self.device)
            vw    = F.softmax(torch.mm(query, hk.t()))
            fact2 = vw.mm(hv_t).mm(drug_memory)
        else:
            fact2 = fact1

        out = self.output(torch.cat([query, fact1, fact2], -1))

        if self.training:
            neg = F.sigmoid(out)
            neg = neg.t() * neg
            return out, neg.mul(self.tensor_ddi_adj).mean()
        return out
