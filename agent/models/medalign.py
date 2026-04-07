"""MedAlign: Multi-modality alignment for medication recommendation.

Architecture must exactly match saved checkpoints from medrec_pipeline training.
Falls back to a GRU-only mode when structure/text PKL files are absent.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

try:
    import ot as pot
    _OT = True
except ImportError:
    _OT = False


def _get_indices(matrix):
    nz   = torch.nonzero(matrix, as_tuple=False)[:, 1]
    lens = list(torch.sum(matrix, 1).long().cpu().numpy())
    return nz, lens


def _get_nonzero(mat):
    return torch.masked_select(mat, mat != 0)


class SubGraphNet(nn.Module):
    def __init__(self, sub_embed, emb_dim, smile_num):
        super().__init__()
        self.sub_emb     = sub_embed
        self.smile_emb   = nn.Embedding(smile_num, emb_dim)
        self.recency_emb = nn.Embedding(30, emb_dim)
        self.degree_emb  = nn.Embedding(30, emb_dim)
        self.dropout     = nn.Dropout(0.7)
        self.ff          = nn.Linear(emb_dim, emb_dim, bias=False)
        enc = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=2, dim_feedforward=emb_dim,
            dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)

    def _get_emb(self, smat, rmat, dmat):
        ids, lens = _get_indices(smat)
        seqs = [v for v in torch.split(ids, lens)]
        p    = pad_sequence(seqs, batch_first=True, padding_value=0).long()
        emb  = self.sub_emb(p)
        mask = (p == 0)
        rec  = _get_nonzero(rmat)
        rec  = [torch.sort(v)[1] for v in torch.split(rec, lens)]
        rec  = pad_sequence(rec, batch_first=True, padding_value=0).long()
        r_e  = self.recency_emb(rec)
        deg  = _get_nonzero(dmat)
        deg  = [torch.sort(v)[1] for v in torch.split(deg, lens)]
        deg  = pad_sequence(deg, batch_first=True, padding_value=0).long()
        d_e  = self.degree_emb(deg)
        return self.dropout(self.ff(r_e + d_e)) + emb, mask

    def forward(self, inputs, query=None):
        smile_sub, rec, deg, drug_smile = inputs
        emb, mask = self._get_emb(smile_sub, rec, deg)
        x   = self.transformer(emb, src_key_padding_mask=mask)
        m   = (~mask).float().unsqueeze(-1)
        rep = (x * m).sum(1) / m.sum(1).clamp(min=1)
        rep = rep + self.smile_emb.weight
        if query is not None:
            a = F.softmax(query.matmul(rep.t()), -1)
            return F.normalize(a.matmul(rep), p=2, dim=1)
        return rep


class MedAlignNet(nn.Module):
    def __init__(self, voc_size, emb_dim, drug_smile, smile_sub,
                 ddi_matrix, structure_matrix, drug_text_embs, device):
        super().__init__()
        self.device  = device
        self.emb_dim = emb_dim
        self.voc_size = voc_size

        self.diag_emb = nn.Embedding(voc_size[0], emb_dim)
        self.proc_emb = nn.Embedding(voc_size[1], emb_dim)
        self.dropout  = nn.Dropout(0.7)
        self.diag_enc = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.proc_enc = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.his_enc  = nn.GRU(emb_dim, emb_dim, batch_first=True)

        self.drug_smile = drug_smile
        self.smile_sub  = smile_sub
        num_drugs = drug_smile.shape[0] if drug_smile is not None else voc_size[2]

        if drug_text_embs is not None:
            text_in = drug_text_embs.shape[-1]
            self.query_plm = nn.Sequential(nn.ReLU(), nn.Linear(text_in, emb_dim))
        else:
            self.query_plm = None
        self.drug_text_embs = drug_text_embs

        sub_n = smile_sub.shape[1] if smile_sub is not None else 1
        smile_n = drug_smile.shape[1] if drug_smile is not None else 1
        self.sub_embed = nn.Embedding(sub_n, emb_dim)
        self.graph_net = SubGraphNet(self.sub_embed, emb_dim, smile_n).to(device)
        self.stru = structure_matrix

        self.drug_emb_id = nn.Embedding(num_drugs, emb_dim)
        self.W1 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W2 = nn.Linear(emb_dim, emb_dim, bias=False)

        num_out = voc_size[2] if voc_size[2] > 0 else num_drugs
        self.drug_mlp = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, num_out)
        )
        self.ddi_matrix = ddi_matrix

        self.flow_s2id: torch.Tensor = None
        self.flow_t2id: torch.Tensor = None

        def _iw(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
        self.apply(_iw)

    def _compute_ot(self):
        if not _OT or self.drug_text_embs is None:
            return
        with torch.no_grad():
            stru = self.graph_net(
                [self.smile_sub, self.stru[0], self.stru[1], self.drug_smile],
                self.drug_emb_id.weight)
            text = self.query_plm(self.drug_text_embs)
            ids  = self.drug_emb_id.weight
            n    = stru.shape[0]
            w    = torch.ones(n, device=self.device) / n
            ns_i = torch.sqrt(torch.norm(stru, "fro")) * torch.sqrt(torch.norm(ids, "fro"))
            nt_i = torch.sqrt(torch.norm(text, "fro")) * torch.sqrt(torch.norm(ids, "fro"))
            C_s  = 1 - stru.mm(ids.t()) / ns_i.item()
            C_t  = 1 - text.mm(ids.t()) / nt_i.item()
            self.flow_s2id = torch.tensor(
                pot.sinkhorn(w.cpu().numpy(), w.cpu().numpy(),
                             C_s.cpu().numpy(), 50, numItermax=1000),
                device=self.device)
            self.flow_t2id = torch.tensor(
                pot.sinkhorn(w.cpu().numpy(), w.cpu().numpy(),
                             C_t.cpu().numpy(), 50, numItermax=1000),
                device=self.device)

    def _get_drug_emb(self):
        if self.flow_s2id is None or not _OT:
            return self.drug_emb_id.weight
        stru = self.graph_net(
            [self.smile_sub, self.stru[0], self.stru[1], self.drug_smile],
            self.drug_emb_id.weight)
        text = self.query_plm(self.drug_text_embs) if self.query_plm else stru
        ot_s = self.flow_s2id.t().mm(stru)
        ot_t = self.flow_t2id.t().mm(text)
        ids  = self.drug_emb_id.weight
        H    = torch.stack([ot_s, ot_t, ids])
        H1   = self.W1(H)
        H2   = self.W2(((ot_s + ot_t + ids) / 3).unsqueeze(0))
        w_c  = F.softmax(
            torch.matmul(H1, H2.transpose(-1, -2)) / math.sqrt(H1.size(-1)), dim=0)
        return (w_c[0].unsqueeze(-1) * ot_s +
                w_c[1].unsqueeze(-1) * ot_t +
                w_c[2].unsqueeze(-1) * ids)

    def forward(self, input_seq):
        drug_emb = self._get_drug_emb()
        d_s, p_s, h_s = [], [], []
        hist = []
        for adm in input_seq:
            if hist:
                he = self.dropout(
                    drug_emb[torch.LongTensor(hist).to(self.device)]
                ).sum(0, keepdim=True).unsqueeze(0)
            else:
                he = torch.zeros(1, 1, self.emb_dim, device=self.device)
            h_s.append(he)
            d_s.append(self.dropout(self.diag_emb(
                torch.LongTensor(adm[0]).unsqueeze(0).to(self.device))).sum(1).unsqueeze(0))
            p_s.append(self.dropout(self.proc_emb(
                torch.LongTensor(adm[1]).unsqueeze(0).to(self.device))).sum(1).unsqueeze(0))
            hist.extend(adm[2])
        d_s = torch.cat(d_s, 1); p_s = torch.cat(p_s, 1); h_s = torch.cat(h_s, 1)
        o1, _ = self.diag_enc(d_s)
        o2, _ = self.proc_enc(p_s)
        o3, _ = self.his_enc(h_s)
        q = torch.cat([o1, o2, o3], -1).squeeze(0)[-1:]   # (1, 3*emb)
        result = self.drug_mlp(q)

        neg = torch.sigmoid(result)
        neg = neg.t() * neg
        ddi = 0.0005 * neg.mul(self.ddi_matrix).sum()
        return result, ddi


class MedAlignFallback(nn.Module):
    """Simple GRU baseline used when structure/text PKL files are absent."""
    def __init__(self, voc_size, emb_dim, ddi_matrix, device):
        super().__init__()
        self.device = device
        self.d_emb  = nn.Embedding(voc_size[0], emb_dim)
        self.p_emb  = nn.Embedding(voc_size[1], emb_dim)
        self.enc_d  = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.enc_p  = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.out    = nn.Linear(2 * emb_dim, voc_size[2])
        self.ddi    = ddi_matrix

    def forward(self, seq):
        ds, ps = [], []
        for adm in seq:
            ds.append(self.d_emb(
                torch.LongTensor(adm[0]).to(self.device)).sum(0, keepdim=True).unsqueeze(0))
            ps.append(self.p_emb(
                torch.LongTensor(adm[1]).to(self.device)).sum(0, keepdim=True).unsqueeze(0))
        o1, _ = self.enc_d(torch.cat(ds, 1))
        o2, _ = self.enc_p(torch.cat(ps, 1))
        q = torch.cat([o1, o2], -1).squeeze(0)[-1:]
        r = self.out(q)
        neg = torch.sigmoid(r); neg = neg.t() * neg
        return r, 0.0005 * neg.mul(self.ddi).sum()
