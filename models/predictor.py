import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch, Data
from utils.chem_utils import smile_to_graph, compute_fingerprint

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.fc(x)

class ResidualRegressor(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            ResidualBlock(512, dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            ResidualBlock(256, dropout),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x)

class DeepInteraction(nn.Module):
    def __init__(self, dim, nhead=4, dropout=0.1):
        super().__init__()
        self.attn_rna_to_mol = nn.MultiheadAttention(dim, nhead, batch_first=True, dropout=dropout)
        self.attn_mol_to_rna = nn.MultiheadAttention(dim, nhead, batch_first=True, dropout=dropout)
        self.norm_rna = nn.LayerNorm(dim)
        self.norm_mol = nn.LayerNorm(dim)

    def forward(self, rna_seq, mol_nodes, rna_mask=None, mol_mask=None):
        rna_key_mask = None if rna_mask is None else ~(rna_mask.bool())
        mol_key_mask = None if mol_mask is None else ~(mol_mask.bool())

        ctx_rna, _ = self.attn_rna_to_mol(query=rna_seq, key=mol_nodes, value=mol_nodes, 
                                          key_padding_mask=mol_key_mask)
        res_rna = self.norm_rna(rna_seq + ctx_rna)
        
        ctx_mol, _ = self.attn_mol_to_rna(query=mol_nodes, key=rna_seq, value=rna_seq, 
                                          key_padding_mask=rna_key_mask)
        res_mol = self.norm_mol(mol_nodes + ctx_mol)
        
        return res_rna, res_mol

class MultiScaleGNN(nn.Module):
    def __init__(self, dim_atom, embed_dim, num_layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.atom_embed = nn.Linear(dim_atom, embed_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            out_channels = embed_dim // heads
            self.convs.append(GATv2Conv(embed_dim, out_channels, heads=heads, edge_dim=6, dropout=dropout))
            self.norms.append(BatchNorm(embed_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.atom_embed(x)
        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.elu(x) + x_in 
        
        h_dense, mol_mask = to_dense_batch(x, batch)
        return h_dense, mol_mask

class AffinityPredictor(nn.Module):
    def __init__(self, dim_atom=80, embed_dim=128, num_heads=8, gnn_layers=3, dropout=0.4, **kwargs):
        super().__init__()
        self.args = kwargs.get('args', {})
        if hasattr(self.args, '__dict__'): self.args = vars(self.args)
        
        self.no_interaction = self.args.get('no_interaction', False)
        self.no_gnn = self.args.get('no_gnn', False)
        self.no_fp = self.args.get('no_fp', False)
        self.no_masking = self.args.get('no_masking', False)

        self.embed_dim = embed_dim
        self.dim_atom = dim_atom
        self.fp_dim = kwargs.get('fp_dim', 2048)
        
        self.rna_proj = nn.Sequential(
            nn.Linear(640, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU()
        )

        gnn_heads = num_heads // 2 if num_heads > 1 else 1
        self.atom_gnn = MultiScaleGNN(self.dim_atom, self.embed_dim, 
                                      num_layers=gnn_layers, heads=gnn_heads, dropout=dropout)
        
        self.fp_proj = nn.Sequential(
            nn.Linear(self.fp_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.interaction = DeepInteraction(self.embed_dim, nhead=num_heads // 2 if num_heads > 1 else 1, dropout=dropout)
        
        reg_input_dim = self.embed_dim
        if not self.no_gnn: reg_input_dim += self.embed_dim
        if not self.no_fp: reg_input_dim += self.embed_dim
        
        self.regressor = ResidualRegressor(reg_input_dim, dropout=dropout)

    def forward(self, batch):
        if not hasattr(batch, 'rna_embedding'):
            raise ValueError("Batch data must contain 'rna_embedding'.")

        rna_seq_raw = batch.rna_embedding.float()
        
        rna_mask = None if self.no_masking else getattr(batch, 'rna_mask', None)
        if rna_mask is not None and rna_mask.dim() == 3:
            rna_mask = rna_mask.squeeze(1)

        rna_seq_feat = self.rna_proj(rna_seq_raw) 
        if rna_seq_feat.dim() == 2:
            rna_seq_feat = rna_seq_feat.unsqueeze(1)
        elif rna_seq_feat.dim() == 4:
            rna_seq_feat = rna_seq_feat.squeeze(1)

        if self.no_gnn:
            mol_nodes_feat, mol_mask = None, None
        else:
            mol_nodes_feat, mol_mask = self.atom_gnn(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            if self.no_masking:
                mol_mask = None

        if self.no_interaction or self.no_gnn:
            refined_rna_seq, refined_mol_nodes = rna_seq_feat, mol_nodes_feat
        else:
            refined_rna_seq, refined_mol_nodes = self.interaction(
                rna_seq_feat, mol_nodes_feat, rna_mask=rna_mask, mol_mask=mol_mask
            )

        if rna_mask is not None:
            m_rna = rna_mask.unsqueeze(-1)
            rna_vec = (refined_rna_seq * m_rna).sum(dim=1) / (m_rna.sum(dim=1) + 1e-8)
        else:
            rna_vec = torch.mean(refined_rna_seq, dim=1)

        concat_list = [rna_vec]

        if not self.no_gnn:
            if mol_mask is not None:
                m_mol = mol_mask.unsqueeze(-1)
                mol_vec = (refined_mol_nodes * m_mol).sum(dim=1) / (m_mol.sum(dim=1) + 1e-8)
            else:
                mol_vec = torch.mean(refined_mol_nodes, dim=1)
            concat_list.append(mol_vec)

        if not self.no_fp:
            drug_fp = batch.drug_fp
            if drug_fp.shape[-1] != self.fp_dim:
                drug_fp = drug_fp[:, :self.fp_dim]
            fp_emb = self.fp_proj(drug_fp)
            if fp_emb.dim() == 3: fp_emb = fp_emb.squeeze(1)
            concat_list.append(fp_emb)
        
        combined_feat = torch.cat(concat_list, dim=-1)
        pred = self.regressor(combined_feat).view(-1)

        if self.training:
            return pred, (rna_vec, mol_vec if not self.no_gnn else None)
        return pred

    @torch.no_grad()
    def predict_smiles_list(self, smiles_list, rna_embedding=None, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        data_list = []
        
        if rna_embedding is None:
            return torch.tensor([-10.0] * len(smiles_list)).to(device)
            
        rna_emb_tensor = rna_embedding if isinstance(rna_embedding, torch.Tensor) else torch.from_numpy(rna_embedding)
        rna_emb_tensor = rna_emb_tensor.to(device).float()

        if rna_emb_tensor.dim() == 1:
            rna_emb_tensor = rna_emb_tensor.unsqueeze(0).unsqueeze(0)
        elif rna_emb_tensor.dim() == 2:
            rna_emb_tensor = rna_emb_tensor.unsqueeze(0)

        rna_mask = None if self.no_masking else torch.ones(1, rna_emb_tensor.size(1)).to(device)

        for smi in smiles_list:
            if not smi: continue
            x, edge_index, edge_attr = smile_to_graph(smi)
            try:
                fp = compute_fingerprint(smi, fp_type="morgan", n_bits=self.fp_dim)
                fp_tensor = torch.FloatTensor(fp).unsqueeze(0) 
            except:
                fp_tensor = torch.zeros(1, self.fp_dim)

            if x is not None:
                data = Data(
                    x=torch.FloatTensor(x),
                    edge_index=torch.LongTensor(edge_index),
                    edge_attr=torch.FloatTensor(edge_attr),
                    rna_embedding=rna_emb_tensor, 
                    rna_mask=rna_mask,
                    drug_fp=fp_tensor
                )
                data_list.append(data)
        
        if not data_list:
            return torch.tensor([-10.0] * len(smiles_list)).to(device)

        batch = Batch.from_data_list(data_list).to(device)
        return self.forward(batch)
