"""SafeDrug: Dual molecular graph encoder with DDI-controlled generation.

Architecture must exactly match saved checkpoints from medrec_pipeline training.
The MPNN branch requires rdkit; falls back to a learned drug-embedding lookup.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem  # noqa: F401 (imported for side-effects)
    _RDKIT = True
except ImportError:
    _RDKIT = False


# ─────────────────────────── MPNN helpers ─────────────────────────────────────

def _create_atoms(mol):
    return [a.GetAtomicNum() - 1 for a in mol.GetAtoms()]


def _create_ijbonddict(mol):
    ij = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ij.setdefault(i, []).append((j, b.GetBondTypeAsDouble()))
        ij.setdefault(j, []).append((i, b.GetBondTypeAsDouble()))
    return ij


def _extract_fingerprints(atoms, ij, radius):
    if radius == 0:
        return atoms
    nodes = atoms[:]
    for _ in range(radius):
        new = []
        for i, a in enumerate(nodes):
            nbrs = tuple(sorted(nodes[j] for j, _ in ij.get(i, [])))
            new.append(hash((a, nbrs)) % (2**16))
        nodes = new
    return nodes


def build_mpnn_inputs(smiles_dict, idx2word, radius=2, device=torch.device("cpu")):
    """Return (MPNNSet, N_fingerprints, average_projection) or None if rdkit absent."""
    if not _RDKIT:
        return None
    fp_set = set()
    all_fps, all_adjs, all_sizes, drug_list = [], [], [], []
    for idx in sorted(idx2word.keys()):
        atc = idx2word[idx]
        fps_d, adjs_d, sizes_d = [], [], []
        for smi in (smiles_dict.get(atc) or []):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            atoms = _create_atoms(mol)
            ij    = _create_ijbonddict(mol)
            fps   = _extract_fingerprints(atoms, ij, radius)
            fp_set.update(fps)
            n   = len(atoms)
            adj = np.zeros((n, n))
            for i, nbrs in ij.items():
                for j, _ in nbrs:
                    adj[i, j] = 1
            fps_d.append(fps); adjs_d.append(adj); sizes_d.append(n)
        all_fps.extend(fps_d); all_adjs.extend(adjs_d); all_sizes.extend(sizes_d)
        drug_list.append(len(fps_d))

    N_fp  = len(fp_set)
    fp_map = {f: i for i, f in enumerate(fp_set)}

    mpnn_set = []
    for fps, adj, sz in zip(all_fps, all_adjs, all_sizes):
        fp_t  = torch.LongTensor([fp_map[f] for f in fps]).to(device)
        adj_t = torch.FloatTensor(adj).to(device)
        mpnn_set.append((fp_t, adj_t, sz))

    n_drugs  = len(drug_list)
    n_col    = sum(drug_list)
    avg_proj = np.zeros((n_drugs, n_col))
    c = 0
    for i, cnt in enumerate(drug_list):
        if cnt > 0:
            avg_proj[i, c:c + cnt] = 1.0 / cnt
        c += cnt
    return mpnn_set, N_fp, torch.FloatTensor(avg_proj).to(device)


# ─────────────────────────── model ────────────────────────────────────────────

class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias   = nn.Parameter(torch.FloatTensor(out_features))
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, mask):
        return x.mm(self.weight * mask) + self.bias


class MPNN(nn.Module):
    def __init__(self, N_fp, emb_dim, n_layers, device):
        super().__init__()
        self.device   = device
        self.n_layers = n_layers
        self.emb      = nn.Embedding(N_fp, emb_dim)
        self.W        = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(n_layers)])

    def forward(self, mpnn_set):
        fps_list, adj_list, sz_list = zip(*mpnn_set)
        fps  = torch.cat(fps_list)
        adjs = self._pad(adj_list)
        h    = self.emb(fps)
        for l in range(self.n_layers):
            h = F.relu(self.W[l](h)) + adjs.mm(F.relu(self.W[l](h)))
        chunks = torch.split(h, list(sz_list))
        return torch.stack([c.sum(0) for c in chunks])

    def _pad(self, mats):
        M = sum(m.shape[0] for m in mats)
        N = sum(m.shape[1] for m in mats)
        out = torch.zeros(M, N, device=self.device)
        r = c = 0
        for m in mats:
            h, w = m.shape
            out[r:r + h, c:c + w] = m
            r += h; c += w
        return out


class SafeDrug(nn.Module):
    def __init__(self, voc_size, ddi_adj, ddi_mask_H, mpnn_data,
                 emb_dim=256, device=torch.device("cpu")):
        """
        mpnn_data: None  → simple drug-embedding fallback
                   tuple → (MPNNSet, N_fingerprints, average_projection)
        """
        super().__init__()
        self.device   = device
        self.voc_size = voc_size

        self.embeddings = nn.ModuleList([
            nn.Embedding(voc_size[i], emb_dim) for i in range(2)])
        self.dropout  = nn.Dropout(0.5)
        self.encoders = nn.ModuleList([
            nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))

        self.tensor_ddi_adj    = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)

        self.bipartite_transform = nn.Linear(emb_dim, ddi_mask_H.shape[1])
        self.bipartite_output    = MaskLinear(ddi_mask_H.shape[1], voc_size[2])

        if mpnn_data is not None:
            mpnn_set, N_fp, avg_proj = mpnn_data
            self.use_mpnn = True
            self.mpnn = MPNN(N_fp, emb_dim, n_layers=2, device=torch.device("cpu"))
            with torch.no_grad():
                mol_emb = self.mpnn(mpnn_set)
                self.MPNN_emb = avg_proj.mm(mol_emb).detach()
                self.MPNN_emb.requires_grad_(True)
            self.MPNN_emb       = nn.Parameter(self.MPNN_emb)
            self.MPNN_output    = nn.Linear(voc_size[2], voc_size[2])
            self.MPNN_layernorm = nn.LayerNorm(voc_size[2])
        else:
            self.use_mpnn = False
            self.drug_emb = nn.Embedding(voc_size[2], emb_dim)
            self.drug_out = nn.Linear(emb_dim, voc_size[2])

        for emb in self.embeddings:
            emb.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input_seq):
        i1, i2 = [], []
        for adm in input_seq:
            d = self.dropout(self.embeddings[0](
                torch.LongTensor(adm[0]).unsqueeze(0).to(self.device))).sum(1).unsqueeze(0)
            p = self.dropout(self.embeddings[1](
                torch.LongTensor(adm[1]).unsqueeze(0).to(self.device))).sum(1).unsqueeze(0)
            i1.append(d); i2.append(p)
        i1 = torch.cat(i1, 1); i2 = torch.cat(i2, 1)
        o1, _ = self.encoders[0](i1)
        o2, _ = self.encoders[1](i2)
        pr = torch.cat([o1, o2], -1).squeeze(0)
        q  = self.query(pr)[-1:]   # (1, emb)

        bipartite = self.bipartite_output(
            F.sigmoid(self.bipartite_transform(q)),
            self.tensor_ddi_mask_H.t())

        if self.use_mpnn:
            mpnn_match = F.sigmoid(q.mm(self.MPNN_emb.t()))
            mpnn_att   = self.MPNN_layernorm(
                mpnn_match + self.MPNN_output(mpnn_match))
            result = torch.mul(bipartite, mpnn_att)
        else:
            idx   = torch.arange(self.voc_size[2], device=self.device)
            de    = self.drug_emb(idx)
            match = F.sigmoid(q.mm(de.t()))
            result = torch.mul(bipartite, match)

        neg = F.sigmoid(result)
        neg = neg.t() * neg
        ddi = 0.0005 * neg.mul(self.tensor_ddi_adj).sum()
        return result, ddi
