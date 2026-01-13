import argparse
import torch
import torch.nn.functional as F

from src.pretrained_2D.model import TokenMAE
from src.pretrained_3D.egnn import EGNN
from collections import OrderedDict

def build_tokenmae_args_from_cli():
    args = argparse.Namespace()

    # ===== flags bạn đưa =====
    args.trans_encoder_layer = 4
    args.trans_decoder_layer = 1
    args.custom_trans = True
    args.transformer_norm_input = True
    args.drop_mask_tokens = True
    args.nonpara_tokenizer = True
    args.gnn_token_layer = 1
    args.loss = "mse"
    args.gnn_type = "gin_v2"
    args.decoder_input_norm = True
    args.eps = 0.5

    # ===== các field TokenMAE dùng trực tiếp nhưng bạn không nêu (giữ default hợp lý) =====
    args.gnn_emb_dim = 300
    args.gnn_dropout = 0.0
    args.gnn_JK = "last"
    args.gnn_activation = "relu"
    args.decoder_jk = "last"
    args.loss_all_nodes = False
    args.zero_mask = False

    # ===== PE config (nếu pretrain không bật PE thì để none) =====
    args.pe_type = "none"
    args.laplacian_norm = "none"
    args.max_freqs = 20
    args.eigvec_norm = "L2"
    args.raw_norm_type = "none"
    args.kernel_times = []
    args.kernel_times_func = "none"
    args.layers = 3
    args.post_layers = 2
    args.dim_pe = 28
    args.phi_hidden_dim = 32
    args.phi_out_dim = 32

    return args

def instantiate_tokenmae_2d(device="cuda"):
    args = build_tokenmae_args_from_cli()

    model = TokenMAE(
        gnn_encoder_layer=5,                 # default theo code của bạn (nếu pretrain khác thì sửa)
        gnn_token_layer=args.gnn_token_layer,
        gnn_decoder_layer=3,                 # default (nếu pretrain khác thì sửa)
        gnn_emb_dim=args.gnn_emb_dim,
        nonpara_tokenizer=args.nonpara_tokenizer,
        gnn_JK=args.gnn_JK,
        gnn_dropout=args.gnn_dropout,
        gnn_type=args.gnn_type,

        d_model=128,                         # default theo add_args
        trans_encoder_layer=args.trans_encoder_layer,
        trans_decoder_layer=args.trans_decoder_layer,
        nhead=4,                             # default theo add_args
        dim_feedforward=512,                 # default theo add_args
        transformer_dropout=0.0,             # default theo add_args
        transformer_activation=F.relu,
        transformer_norm_input=args.transformer_norm_input,
        custom_trans=args.custom_trans,
        drop_mask_tokens=args.drop_mask_tokens,

        use_trans_decoder=False,             # bạn không bật --use_trans_decoder
        pe_type=args.pe_type,
        args=args,
    ).to(device)

    return model

def load_tokenmae_checkpoint(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    assert isinstance(sd, OrderedDict), f"Expected OrderedDict state_dict, got {type(sd)}"

    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(f"[TokenMAE-2D] strict=True loaded | missing={len(missing)} unexpected={len(unexpected)}")
    return model

def strip_prefix(sd, prefix: str):
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    out = OrderedDict()
    for k, v in sd.items():
        out[k[len(prefix):]] = v
    return out

def load_egnn_3d(ckpt_path, device):
    model = EGNN(
        in_node_nf=64,
        hidden_nf=128,
        out_node_nf=128,
        in_edge_nf=13,      # bond features = 13
        device=device,
        n_layers=4,
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
    ).to(device)

    sd = torch.load(ckpt_path, map_location="cpu")
    sd = strip_prefix(sd, "module.")
    sd = strip_prefix(sd, "egnn.")     # nếu checkpoint lưu dạng wrapper 'egnn.xxx'

    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(f"[3D EGNN] loaded strict=True | missing={len(missing)} unexpected={len(unexpected)}")

    # Sanity check: input dim edge_mlp phải là 270
    in_features = model._modules["gcl_0"].edge_mlp[0].weight.shape[1]
    print(f"[3D EGNN] gcl_0.edge_mlp[0] in_features = {in_features} (expect 270)")
    return model


device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_2d = "/path/to/tokenmae_2d_pretrained.pt"
ckpt_3d = "/path/to/egnn_3d_pretrained.pt"

tokenmae_2d = instantiate_tokenmae_2d(device=device)
tokenmae_2d = load_tokenmae_checkpoint(tokenmae_2d, ckpt_2d)

egnn_3d = load_egnn_3d(ckpt_3d, device)

def freeze(m):
    for p in m.parameters():
        p.requires_grad = False

freeze(tokenmae_2d)
freeze(egnn_3d)

print("Frozen both pretrained encoders (ready for Phase-1: train fusion/head).")

