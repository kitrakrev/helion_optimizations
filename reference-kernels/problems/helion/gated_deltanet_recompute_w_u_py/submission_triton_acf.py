#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius

# Triton + embedded ACF (base64). Use if TileIR fails on KernelBot.
# TileIR is ~35% faster locally; this is backup.
from task import input_t, output_t

import base64
import tempfile
from pathlib import Path

import torch
import helion
import helion.language as hl

_ACF_B64 = [
    "dxWiJeTK18brsHzCEqyUfsG2eoBthCEcv4NjjksNymWQxD8cU0RHVJJzMLQZiOz63XaZNYSLXok3djjkTO2n6Nn67r2HJaVwrjKLcuCbrgpqZqMuwKiGX7RRmC14HluHL1c4xLb3bK2LKIhXfpeVII+P1qia2+ItMFi4gE7VSFen9XmiJBIGMcUbNH5beyaknUIjHrgVoEa2BS3LEz55RvWoM22++qrKBO6BmqwBrejxBOb19ibPySaNYH339k8529dbfanIqYyHz3Kgc7KlAKfwXeM=",
    "dxWiJeYWsl07kkfpP9XX/H8eAH47Ex9AMGrP2GEhFvkMWKOAz9jbyA7vrCiFFHCddWvjdjhCfC+RcLvHYrp75TSDqI34ykyBXwb8QV+bTLVnv2jgxaWWBb8H8Cmi26tKOccrQTO4DanmReU6E/r4TRH+2MyPO+IfibkdciisIvf6PUfFlTc=",
    "dxWiJZNK0D9Z8CWLXbe1nh18YhxZcceY6LIXALn5Pv9YfJxbrAZLcViMHYb1sA6Bn/FZhobefsGleab8EL0cTQ1f3KCttadrN0O5QC4v5EBdyEvvZjploski6+PfHPlehPoayw+MH80DoP1c0TrkrLYjSUcEVfzgyTF1/EjTyqSLdIIAJLwg58z9ogOtg09qSKb52dUPMqQ/LJY+keL1eiFm45g0oeiO5NDc8xp1XEIKPmixbg/mvuJkqN7saj+CNNgusLuRsnmiu4vrpi4YvZzpZEEE87iPf5lJ1U7gNcuhuYDTfYqgb1JSbnqvMpLMlZfpEjs6ATMnsX3IgzJB7Ti119xYQObBnhPnOFuuFKi3/5nol9nzS+A/WgThyq2zjX4cA0SmsM30lvcUozXJiMH9qJTkSBuDt7JlU3ksCje1+tciI85jEWzgmi5+eGRGhwAQ3CfxA4+JIYmEoolO7J0Ip1GMJTerufjf8vFNx4K7OwP3cU+1cDtF1w8mAkkV7ez3wHn4pt845UPAUqeU94JVP7B3GbAJyHt0RuPeWfPyVRX/tR14k3Qh",
    "dxWiJeYZcZ74UYQq/BYUP7zdw7340NyC6WICRzM4T9bjx147O2/Caw2RhgKbKfhoehTGUx1nYcbq6h1HpEupdbze4ivd7/EU7MDsFdkJsvgqKCP5GkYRyDlSEstIT1kfonlBAF7d7uyQXDBJSgl4MC33rmCL3uNKKWOPiNnk+JbEQMxHE7HbjObXgEMWNXKO87nb00YmCCHpWupG6U8IN4TZQhzPHSMh7wVqcUfqRgMa7w0eHc0kA93iZg6OxnKLAgVbAQ+45j2+QwnTDNTMaUWBWwo4RnQ=",
    "dxWiJfsEFfqcNeBOmHJwW9i5p9mctPDTmm6OPso=",
]


def _ensure_acf_dir():
    if hasattr(_ensure_acf_dir, "_acf_paths"):
        return _ensure_acf_dir._acf_paths
    d = tempfile.mkdtemp(prefix="recompute_w_u_acf_")
    paths = [str(Path(d) / f"recompute_w_u_fwd_{i}.acf") for i in range(len(_ACF_B64))]
    for i, b64 in enumerate(_ACF_B64):
        Path(paths[i]).write_bytes(base64.b64decode(b64))
    _ensure_acf_dir._acf_paths = paths
    return paths


_BEST_ACF = _ensure_acf_dir()[0]

SHAPE_CONFIGS = {
    s: helion.Config(advanced_controls_file=_BEST_ACF, block_sizes=[], num_warps=4, num_stages=2)
    for s in [
        (1, 64, 2, 64, 64), (2, 128, 4, 64, 64), (1, 256, 4, 64, 128),
        (1, 64, 1, 64, 64), (2, 512, 3, 64, 64), (2, 1024, 3, 64, 64),
        (3, 1024, 4, 100, 100), (4, 1024, 4, 128, 128), (2, 1536, 4, 128, 128), (4, 2048, 8, 64, 64),
    ]
}


def _make_kernel(config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(k, v, beta, A, g):
        B, T, H, K = k.shape
        V = v.shape[-1]
        C = hl.specialize(A.shape[-1])
        K, V = hl.specialize(K), hl.specialize(V)
        w_out, u_out = torch.empty_like(k), torch.empty_like(v)
        BH = B * H
        for flat_bh, rt in hl.tile([BH, T], block_size=[1, C]):
            b_idx, h_idx = flat_bh.begin // H, flat_bh.begin % H
            A_chunk = A[b_idx, rt, h_idx, :].to(torch.float32)
            k_chunk = k[b_idx, rt, h_idx, :].to(torch.float32)
            v_chunk = v[b_idx, rt, h_idx, :].to(torch.float32)
            beta_chunk = beta[b_idx, rt, h_idx].to(torch.float32)
            g_chunk = g[b_idx, rt, h_idx].to(torch.float32)
            scaled_k = k_chunk * (beta_chunk * torch.exp(g_chunk))[:, None]
            scaled_v = v_chunk * beta_chunk[:, None]
            w_out[b_idx, rt, h_idx, :] = hl.dot(A_chunk, scaled_k, out_dtype=torch.float32).to(k.dtype)
            u_out[b_idx, rt, h_idx, :] = hl.dot(A_chunk, scaled_v, out_dtype=torch.float32).to(v.dtype)
        return w_out, u_out
    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    return _KERNELS[(k.shape[0], k.shape[1], k.shape[2], k.shape[3], v.shape[-1])](k, v, beta, A, g)
