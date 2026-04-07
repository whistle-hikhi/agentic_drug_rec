"""MoleRec: Substructure-aware molecular recommendation.

Uses embedding-table mode (no DGL/GNN required).
Architecture must exactly match saved checkpoints from medrec_pipeline training.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdjAttenAggr(nn.Module):
    """Substructure-to-molecule aggregation via masked attention."""
    def __init__(self, Qdim, Kdim, mid_dim):
        super().__init__()
        self.Qdense = nn.Linear(Qdim, mid_dim)
        self.Kdense = nn.Linear(Kdim, mid_dim)
        self.scale  = math.sqrt(mid_dim)

    def forward(self, main_feat, other_feat, fix_feat, mask=None):
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        A = torch.matmul(Q, K.t()) / self.scale
        if mask is not None:
            A = A.masked_fill(mask, -(1 << 32))
        A = torch.softmax(A, -1)
        F_ = torch.diag(fix_feat)
        return A.mm(F_.mm(other_feat))


class MoleRec(nn.Module):
    def __init__(self, voc_size, substruct_num, emb_dim=64,
                 dropout=0.7, device=torch.device("cpu")):
        super().__init__()
        self.device = device

        self.embeddings = nn.ModuleList([
            nn.Embedding(voc_size[0], emb_dim),
            nn.Embedding(voc_size[1], emb_dim),
        ])
        self.seq_encoders = nn.ModuleList([
            nn.GRU(emb_dim, emb_dim, batch_first=True),
            nn.GRU(emb_dim, emb_dim, batch_first=True),
        ])
        self.rnn_drop = nn.Dropout(dropout)

        self.substruct_emb  = nn.Parameter(torch.zeros(substruct_num, emb_dim))
        nn.init.xavier_uniform_(self.substruct_emb.unsqueeze(0))

        self.query          = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim * 4, emb_dim))
        self.substruct_rela = nn.Linear(emb_dim, substruct_num)

        self.sab = nn.MultiheadAttention(emb_dim, num_heads=2, batch_first=True)

        self.aggregator      = AdjAttenAggr(emb_dim, emb_dim, emb_dim)
        self.score_extractor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2), nn.ReLU(),
            nn.Linear(emb_dim // 2, 1),
        )

        self.drug_global = nn.Embedding(voc_size[2], emb_dim)

        for e in self.embeddings:
            e.weight.data.uniform_(-0.1, 0.1)
        self.drug_global.weight.data.uniform_(-0.1, 0.1)

    def forward(self, patient_data, ddi_mask_H, tensor_ddi_adj):
        seq1, seq2 = [], []
        for adm in patient_data:
            e1 = self.rnn_drop(self.embeddings[0](
                torch.LongTensor(adm[0]).to(self.device))).sum(0, keepdim=True)
            e2 = self.rnn_drop(self.embeddings[1](
                torch.LongTensor(adm[1]).to(self.device))).sum(0, keepdim=True)
            seq1.append(e1.unsqueeze(0))
            seq2.append(e2.unsqueeze(0))
        seq1 = torch.cat(seq1, 1); seq2 = torch.cat(seq2, 1)

        out1, h1 = self.seq_encoders[0](seq1)
        out2, h2 = self.seq_encoders[1](seq2)
        seq_repr  = torch.cat([h1, h2], -1)
        last_repr = torch.cat([out1[:, -1], out2[:, -1]], -1)
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])

        query       = self.query(patient_repr)
        substruct_w = torch.sigmoid(self.substruct_rela(query))

        sub_emb, _ = self.sab(
            self.substruct_emb.unsqueeze(0),
            self.substruct_emb.unsqueeze(0),
            self.substruct_emb.unsqueeze(0),
        )
        sub_emb = sub_emb.squeeze(0)

        drug_idx   = torch.arange(self.drug_global.weight.shape[0], device=self.device)
        global_emb = self.drug_global(drug_idx)

        mask    = torch.logical_not(ddi_mask_H > 0)
        mol_emb = self.aggregator(global_emb, sub_emb, substruct_w, mask=mask)

        score = self.score_extractor(mol_emb).t()

        neg = torch.sigmoid(score)
        neg = neg.t().mm(neg)
        ddi = 0.0005 * neg.mul(tensor_ddi_adj).sum()
        return score, ddi
