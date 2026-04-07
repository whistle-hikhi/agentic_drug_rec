"""RETAIN: Reverse Time Attention for EHR.

Architecture must exactly match saved checkpoints from medrec_pipeline training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Retain(nn.Module):
    def __init__(self, voc_size, emb_size=64, device=torch.device("cpu")):
        super().__init__()
        self.device    = device
        self.voc_size  = voc_size
        self.input_len = voc_size[0] + voc_size[1] + voc_size[2]
        self.output_len = voc_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, emb_size, padding_idx=self.input_len),
            nn.Dropout(0.5),
        )
        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru  = nn.GRU(emb_size, emb_size, batch_first=True)
        self.alpha_li  = nn.Linear(emb_size, 1)
        self.beta_li   = nn.Linear(emb_size, emb_size)
        self.output    = nn.Linear(emb_size, self.output_len)

    def forward(self, input_seq):
        max_len = max(len(v[0]) + len(v[1]) + len(v[2]) for v in input_seq)
        input_np = []
        for v in input_seq:
            row = (v[0]
                   + [c + self.voc_size[0] for c in v[1]]
                   + [c + self.voc_size[0] + self.voc_size[1] for c in v[2]])
            row += [self.input_len] * (max_len - len(row))
            input_np.append(row)

        visit_emb = self.embedding(torch.LongTensor(input_np).to(self.device))
        visit_emb = visit_emb.sum(dim=1)                  # (visits, emb)
        g, _ = self.alpha_gru(visit_emb.unsqueeze(0))
        h, _ = self.beta_gru(visit_emb.unsqueeze(0))
        g = g.squeeze(0); h = h.squeeze(0)
        alpha = F.softmax(self.alpha_li(g), dim=0)        # (visits, 1)
        beta  = torch.tanh(self.beta_li(h))               # (visits, emb)
        c = (alpha * beta * visit_emb).sum(0, keepdim=True)  # (1, emb)
        return self.output(c)
