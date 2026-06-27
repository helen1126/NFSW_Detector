from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from models.layers import GraphConvolution, DistanceAdj

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=torch.bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(
            x, x, x, need_weights=False,
            key_padding_mask=padding_mask,
            attn_mask=self.attn_mask
        )[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class ShotDensityHead(nn.Module):
    def __init__(self, dim: int, hidden: int = None, pi_floor: float = 0.1):
        super().__init__()
        h = hidden or max(dim // 2, 64)
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            QuickGELU(),
            nn.Linear(h, 1)
        )
        with torch.no_grad():
            if isinstance(self.net[-1], nn.Linear):
                nn.init.constant_(self.net[-1].bias, -1.38629436112)

        self.register_buffer(
            "pi_floor",
            torch.tensor(float(max(0.0, min(0.5, pi_floor))))
        )

    def forward(self, shot_tokens: torch.Tensor):
        pi = torch.sigmoid(self.net(shot_tokens)).squeeze(-1)
        eps = 1e-4
        pi = pi.clamp(
            min=float(self.pi_floor.item()) + eps,
            max=1.0 - eps
        )
        return pi

class CFATextAdapter(nn.Module):
    def __init__(self, d_vis, d_txt, num_heads=8, prefix_len=64, bottleneck=256,
                 prefix_rank=16, dropout=0.1,
                 tau_gate: float = 1.0,
                 beta_fuse: float = 1.0
                 ):
        super().__init__()
        self.d_vis = d_vis
        self.d_txt = d_txt
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.prefix_rank = prefix_rank

        self.proj_k = nn.Linear(d_txt, d_vis)
        self.proj_v = nn.Linear(d_txt, d_vis)

        self.ln_q  = nn.LayerNorm(d_vis)
        self.ln_kv = nn.LayerNorm(d_vis)
        self.ln_out = nn.LayerNorm(d_vis)

        self.prefix_k = nn.Parameter(torch.zeros(prefix_len, d_vis))
        self.prefix_v = nn.Parameter(torch.zeros(prefix_len, d_vis))

        self.pool_ln = nn.LayerNorm(d_vis)
        self.prefix_gen_k_in = nn.Linear(d_vis, prefix_len * prefix_rank)
        self.prefix_gen_v_in = nn.Linear(d_vis, prefix_len * prefix_rank)
        self.prefix_out_k = nn.Linear(prefix_rank, d_vis, bias=False)
        self.prefix_out_v = nn.Linear(prefix_rank, d_vis, bias=False)
        self.prefix_drop = nn.Dropout(dropout)

        self.xattn = nn.MultiheadAttention(d_vis, num_heads, dropout=dropout)

        self.down = nn.Linear(d_vis, bottleneck)
        self.up   = nn.Linear(bottleneck, d_vis)
        self.act  = QuickGELU()
        self.drop = nn.Dropout(dropout)

        self.w_mod = nn.Linear(d_vis, d_vis)

        self.fuse_fc = nn.Linear(d_vis, d_vis)

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07, dtype=torch.float32)))

        self.register_buffer("tau_gate", torch.tensor(float(tau_gate)))
        self.register_buffer("beta_fuse", torch.tensor(float(beta_fuse)))

    def _make_dynamic_prefix(self, vis_feat: torch.Tensor):
        B, T, Dv = vis_feat.shape
        h = vis_feat.mean(dim=1)
        h = self.pool_ln(h)

        pk = self.prefix_gen_k_in(h).view(B, self.prefix_len, self.prefix_rank)
        pv = self.prefix_gen_v_in(h).view(B, self.prefix_len, self.prefix_rank)
        pk = self.prefix_out_k(pk)
        pv = self.prefix_out_v(pv)
        pk = self.prefix_drop(pk)
        pv = self.prefix_drop(pv)
        return pk, pv

    def forward(self, vis_feat, txt_feat):
        B, T, Dv = vis_feat.shape
        C, Dt = txt_feat.shape

        K_base = self.ln_kv(self.proj_k(txt_feat))
        V_base = self.ln_kv(self.proj_v(txt_feat))

        Pk_dyn, Pv_dyn = self._make_dynamic_prefix(vis_feat)
        Pk = self.prefix_k.unsqueeze(0).expand(B, -1, -1) + Pk_dyn
        Pv = self.prefix_v.unsqueeze(0).expand(B, -1, -1) + Pv_dyn

        Q = self.ln_q(vis_feat).permute(1, 0, 2)
        K_expand = K_base.unsqueeze(0).expand(B, -1, -1)
        V_expand = V_base.unsqueeze(0).expand(B, -1, -1)
        K_all = torch.cat([Pk, K_expand], dim=1).permute(1, 0, 2).contiguous()
        V_all = torch.cat([Pv, V_expand], dim=1).permute(1, 0, 2).contiguous()
        F_att, _ = self.xattn(Q, K_all, V_all, need_weights=False)
        F_att = F_att.permute(1, 0, 2)

        F_hat = self.up(self.drop(self.act(self.down(F_att))))

        Qn = F.normalize(vis_feat, dim=-1, eps=1e-6)
        Kn = F.normalize(K_expand, dim=-1, eps=1e-6)
        scores = torch.matmul(Qn, Kn.transpose(1, 2))

        tau = self.tau_gate.clamp(min=0.25, max=4.0)
        scores = scores / tau

        weights = torch.softmax(scores, dim=-1)
        F_txt_ctx = torch.matmul(weights, V_expand)

        F_mod = torch.sigmoid(self.w_mod(self.ln_kv(F_txt_ctx)))

        beta = self.beta_fuse.clamp(0.0, 1.0)
        F_fused = self.ln_out(self.fuse_fc(vis_feat + beta * (F_hat * F_mod)))

        Tproj = F.normalize(K_base, dim=-1, eps=1e-6)
        fused_n = F.normalize(F_fused, dim=-1, eps=1e-6)
        logit_scale = self.logit_scale.clamp(max=math.log(20.0)).exp()
        logits2_vis = torch.matmul(fused_n, Tproj.t()) * logit_scale

        return logits2_vis, F_fused

class TextCFAdapter(nn.Module):
    def __init__(self,
                 d_vis: int,
                 d_txt: int,
                 bottleneck: int = 256,
                 dropout: float = 0.1,
                 beta_fuse: float = 1.0):
        super().__init__()
        self.d_vis = d_vis
        self.d_txt = d_txt

        self.proj_t = nn.Linear(d_txt, d_vis)

        self.pool_ln = nn.LayerNorm(d_vis)

        self.mlp = nn.Sequential(
            nn.Linear(d_vis, bottleneck),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, d_vis * 2)
        )

        self.ln_out = nn.LayerNorm(d_vis)

        self.register_buffer("beta_fuse", torch.tensor(float(beta_fuse)))

    def forward(self, txt_feat: torch.Tensor, vis_feat: torch.Tensor):
        C, Dt = txt_feat.shape
        B, T, Dv = vis_feat.shape

        T_base = self.proj_t(txt_feat)

        g = vis_feat.mean(dim=1)
        g = self.pool_ln(g)

        h = self.mlp(g)
        gamma, bias = h.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        beta = bias

        beta_f = self.beta_fuse.clamp(0.0, 1.0)
        gamma = beta_f * gamma
        beta = beta_f * beta

        T_base_b = T_base.unsqueeze(0)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        T_cfa = self.ln_out(T_base_b * (1.0 + gamma) + beta)

        return T_base, T_cfa

class SVLA(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device,
                 shot_sim_thresh: float = 0.88,
                 shot_min_len: int = 3,
                 shot_layers: int = 1,
                 shot_gamma: float = 0.15,
                 pi_floor: float = 0.10,
                 cfa_tau: float = 1.0,
                 cfa_beta: float = 1.0,
                 cfa_prefix_len: int = 64,
                 cfa_bottleneck: int = 256,
                 cfa_prefix_rank: int = 16,
                 cfa_dropout: float = 0.1,
                 clip_variant: str = "ViT-B/16"):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device
        self.clip_variant = clip_variant

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        self.shot_sim_thresh = shot_sim_thresh
        self.shot_min_len = shot_min_len
        self.shot_gamma = shot_gamma
        self.shot_transformer = Transformer(
            width=visual_width,
            layers=shot_layers,
            heads=visual_head,
            attn_mask=None
        )
        self.shot_proj = nn.Linear(visual_width, visual_width)
        self.shot_density_head = ShotDensityHead(visual_width, pi_floor=pi_floor)

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load(self.clip_variant, device)
        for p in self.clipmodel.parameters():
            p.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.cfa_text = CFATextAdapter(
            d_vis=self.visual_width,
            d_txt=self.embed_dim,
            num_heads=visual_head,
            prefix_len=cfa_prefix_len,
            bottleneck=max(cfa_bottleneck, self.visual_width // 2),
            prefix_rank=cfa_prefix_rank,
            dropout=cfa_dropout,
            tau_gate=cfa_tau,
            beta_fuse=cfa_beta
        )

        self.text_cfa = TextCFAdapter(
            d_vis=self.visual_width,
            d_txt=self.embed_dim,
            bottleneck=max(cfa_bottleneck, self.visual_width // 2),
            dropout=cfa_dropout,
            beta_fuse=cfa_beta
        )

        self.lambda_v = nn.Parameter(torch.tensor(0.0))
        self.lambda_t = nn.Parameter(torch.tensor(0.0))

        self._dbg_adj_nz_ratio = None
        self._dbg_pi_mean = None
        self._dbg_pi_min = None
        self._dbg_pi_max = None
        self._last_shot_pi_list = None

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        L = self.visual_length
        mask = torch.ones(L, L, dtype=torch.bool)
        n_blocks = (L + attn_window - 1) // attn_window
        for i in range(n_blocks):
            s = i * attn_window
            e = min((i + 1) * attn_window, L)
            mask[s:e, s:e] = False
        return mask

    @torch.no_grad()
    def _detect_shots(self, feats: torch.Tensor, lengths: torch.Tensor):
        B, T, D = feats.shape
        feats_n = F.normalize(feats, dim=-1, eps=1e-6)
        shot_slices = []
        for b in range(B):
            L = int(lengths[b].item())
            if L <= 1:
                shot_slices.append([(0, max(L, 1))])
                continue
            adj_sim = (feats_n[b, :L-1] * feats_n[b, 1:L]).sum(dim=-1)
            cut_idx = (adj_sim < self.shot_sim_thresh).nonzero(as_tuple=False).reshape(-1).tolist()
            bounds = [0]
            for c in cut_idx:
                bounds.append(c + 1)
            bounds.append(L)

            merged, cur_s = [], bounds[0]
            for idx in range(1, len(bounds)):
                cur_e = bounds[idx]
                if cur_e - cur_s < self.shot_min_len and idx < len(bounds) - 1:
                    continue
                merged.append((cur_s, cur_e))
                cur_s = cur_e
            if len(merged) == 0:
                merged = [(0, L)]
            shot_slices.append(merged)
        return shot_slices

    def _adjacency_shotaware(self, x, seq_len, shot_slices=None, gamma=0.15, thr=0.7):
        x2 = x @ x.transpose(1, 2)
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)
        x2 = x2 / (x_norm @ x_norm.transpose(1, 2) + 1e-20)

        B, T, _ = x.shape
        A = torch.zeros_like(x2)

        for i in range(B):
            Li = int(seq_len[i].item())
            if Li <= 0:
                continue
            A_sim = F.threshold(x2[i, :Li, :Li], thr, 0.0)

            if shot_slices is not None:
                A_shot = torch.zeros_like(A_sim)
                for (s, e) in shot_slices[i]:
                    s, e = max(0, s), min(Li, e)
                    if e > s:
                        A_shot[s:e, s:e] = 1.0
                A_sim = A_sim + gamma * A_shot

            Ai = A_sim + torch.eye(Li, device=x.device)
            d = Ai.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            D_inv_sqrt = d.pow(-0.5)
            A[i, :Li, :Li] = D_inv_sqrt * Ai * D_inv_sqrt.transpose(0, 1)

        A[A != A] = 0
        return A

    @staticmethod
    def _mask_row_normalize(adj: torch.Tensor, lengths: torch.Tensor):
        B, T, _ = adj.shape
        masks = torch.ones(B, T, T, device=adj.device)
        for i in range(B):
            Li = int(lengths[i].item())
            if Li < T:
                masks[i, Li:, :] = 0
                masks[i, :, Li:] = 0
        adj_masked = adj * masks
        row_sum = adj_masked.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adj_masked / row_sum

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)

        B, T, D = images.shape
        if lengths is not None:
            frame_padding_mask = torch.ones(B, T, dtype=torch.bool, device=images.device)
            for i in range(B):
                Li = int(lengths[i].item())
                frame_padding_mask[i, :Li] = False
        else:
            frame_padding_mask = None

        x = images + frame_position_embeddings
        x = x.permute(1, 0, 2)
        x, _ = self.temporal((x, frame_padding_mask))
        x = x.permute(1, 0, 2)

        with torch.no_grad():
            shot_slices = self._detect_shots(x, lengths)

        shot_tokens_list, max_shots = [], 0
        for b in range(B):
            tokens = [x[b, s:e].mean(dim=0) for (s, e) in shot_slices[b] if e > s]
            if len(tokens) == 0:
                Li = int(lengths[b].item())
                tokens = [x[b, :Li].mean(dim=0)]
            max_shots = max(max_shots, len(tokens))
            shot_tokens_list.append(tokens)

        shot_tokens = x.new_zeros(B, max_shots, D)
        shot_pad_mask = torch.ones(B, max_shots, dtype=torch.bool, device=x.device)
        for b in range(B):
            m = len(shot_tokens_list[b])
            shot_tokens[b, :m] = torch.stack(shot_tokens_list[b], dim=0)
            shot_pad_mask[b, :m] = False

        st = shot_tokens.permute(1, 0, 2)
        st, _ = self.shot_transformer((st, shot_pad_mask))
        st = st.permute(1, 0, 2)
        st = self.shot_proj(st)

        shot_pi = self.shot_density_head(st)
        shot_pi_list = []
        for b in range(B):
            m_valid = (~shot_pad_mask[b]).sum().item()
            if m_valid > 0:
                shot_pi_list.append(shot_pi[b, :m_valid])
            else:
                shot_pi_list.append(torch.sigmoid(st[b, :1, 0]))
        self._last_shot_pi_list = shot_pi_list

        with torch.no_grad():
            valid = ~shot_pad_mask
            if valid.any():
                vals = shot_pi[valid]
                self._dbg_pi_mean = float(vals.mean().item())
                self._dbg_pi_min  = float(vals.min().item())
                self._dbg_pi_max  = float(vals.max().item())
            else:
                self._dbg_pi_mean = self._dbg_pi_min = self._dbg_pi_max = None

        adj = self._adjacency_shotaware(x, lengths, shot_slices=shot_slices,
                                        gamma=self.shot_gamma, thr=0.7)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        disadj = self._mask_row_normalize(disadj, lengths)

        with torch.no_grad():
            ratios = []
            for i in range(B):
                Li = int(lengths[i].item())
                if Li <= 0:
                    continue
                sub = adj[i, :Li, :Li]
                nz = (sub.abs() > 1e-12).float().sum()
                ratios.append((nz / (Li * Li)).unsqueeze(0))
            self._dbg_adj_nz_ratio = torch.cat(ratios).mean().item() if len(ratios) > 0 else None

        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))
        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), dim=2)
        x = self.linear(x)
        x = x + self.mlp1(x)

        return x, shot_slices

    def encode_textprompt(self, text):
        tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(tokens)
        N, L = tokens.size(0), tokens.size(1)

        base = self.text_prompt_embeddings(
            torch.arange(L, device=self.device)
        ).unsqueeze(0).repeat(N, 1, 1)
        text_tokens = torch.zeros(N, L, device=self.device)

        try:
            eot_id = self.clipmodel.tokenizer.eot
        except Exception:
            eot_id = 49407

        eot_pos = (tokens == eot_id).float().argmax(dim=-1)

        for i in range(N):
            ind = int(eot_pos[i].item())
            ind = max(1, min(ind, L - 1))
            start = max(1, min(self.prompt_prefix + 1, L - 1))
            end   = max(start, min(self.prompt_prefix + ind, L - 1))
            eot_tgt = min(self.prompt_prefix + ind + self.prompt_postfix, L - 1)

            base[i, 0] = word_embedding[i, 0]

            copy_len = min(end - start, ind - 1)
            if copy_len > 0:
                base[i, start:start + copy_len] = word_embedding[i, 1:1 + copy_len]

            base[i, eot_tgt] = word_embedding[i, ind]
            text_tokens[i, eot_tgt] = tokens[i, ind]

        text_features = self.clipmodel.encode_text(base, text_tokens)
        return text_features

    def forward(self, visual, padding_mask, text, lengths):
        visual_features, shot_slices = self.encode_video(visual, padding_mask, lengths)

        text_features_ori = self.encode_textprompt(text)

        _logits2_vis, V_cfa = self.cfa_text(visual_features, text_features_ori)

        T_base_vis, T_cfa_vis = self.text_cfa(text_features_ori, visual_features)

        lambda_v = torch.sigmoid(self.lambda_v)
        lambda_t = torch.sigmoid(self.lambda_t)

        V_f = (1.0 - lambda_v) * visual_features + lambda_v * V_cfa

        T_base_b = T_base_vis.unsqueeze(0)
        T_f = (1.0 - lambda_t) * T_base_b + lambda_t * T_cfa_vis

        V_norm = F.normalize(V_f, dim=-1, eps=1e-6)
        T_norm = F.normalize(T_f, dim=-1, eps=1e-6)
        T_norm = T_norm.permute(0, 2, 1)

        logit_scale = self.cfa_text.logit_scale.clamp(max=math.log(20.0)).exp()
        logits2 = torch.matmul(V_norm, T_norm) * logit_scale

        feat_bin = visual_features + self.mlp2(visual_features)
        logits1 = self.classifier(feat_bin)

        visual_features_norm = feat_bin / (feat_bin.norm(dim=-1, keepdim=True) + 1e-12)

        return text_features_ori, logits1, logits2, shot_slices, visual_features_norm
