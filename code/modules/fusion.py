import torch
from torch import nn 
import math
from einops import repeat
from .selective_scan_compat import selective_scan_fn
import torch.nn.functional as F
from einops import rearrange
from .cssc import DynamicScaleGate, cssc_gate_values
from .boundary import BoundaryReliabilityEstimator
"""
u: (B D L)
Delta: (B D L) Aware Private
A: (D H) Independent Common
B: (B H L) Aware Private
C: (B H L) Aware Private
D: (D) Independent Common
h'(t) = Ah(t) + Bx(t)
y(t)  = Ch(t) + Dx(t)
"""

d_1, d_2 = 0, 0
c_1, c_2 = 0, 0

class FusionBlock(nn.Module):
    def __init__(self, d_model, d_state=32, d_conv=3, expand=1.5, dt_rank="auto", dt_min=0.001,
            dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, dropout=0.,
                conv_bias=True, bias=False, device=None, dtype=None, global_scales=(),
                global_op="scan", gate_type="scalar", dynamic_gate_hidden=32,
                boundary_mode="none", boundary_hidden=24,
                boundary_global_context=True, **kwargs,) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # x represent infrared images
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        # y represent visible images
        self.y_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.y_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.y_proj], dim=0)) # (K=4, N, inner)
        del self.y_proj

        # shared weight of dt_projs
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank) 
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

                
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N) # Context Independent
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.mark_proj = nn.Parameter(nn.Linear(self.d_inner * 2, self.d_inner, bias=False).weight)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # Multi-scale global branch (same idea as SS2D): downsample, Cross-SS2D, upsample, gated fusion.
        self.global_scales = tuple(int(s) for s in (global_scales or ()))
        self.global_gates = nn.Parameter(torch.zeros(len(self.global_scales))) if self.global_scales else None
        self.global_op = str(global_op).lower()
        if self.global_op not in {"scan", "pool1x1", "dw3x3"}:
            raise ValueError(f"Unsupported global_op={global_op}, expected one of scan/pool1x1/dw3x3")
        self.gate_type = str(gate_type).lower()
        if self.gate_type not in {"scalar", "dynamic"}:
            raise ValueError(f"Unsupported gate_type={gate_type}, expected scalar/dynamic")
        self.dynamic_gate = (
            DynamicScaleGate(self.d_inner * 2, self.global_scales, hidden=dynamic_gate_hidden)
            if self.global_scales and self.gate_type == "dynamic"
            else None
        )
        self.boundary_mode = str(boundary_mode).lower()
        if self.boundary_mode not in {"none", "residual", "state", "full"}:
            raise ValueError("boundary_mode must be one of none/residual/state/full")
        self.boundary_estimator = (
            BoundaryReliabilityEstimator(
                self.d_inner,
                hidden=boundary_hidden,
                use_global_context=boundary_global_context,
            )
            if self.boundary_mode != "none"
            else None
        )
        # Zero initialization keeps a loaded S4Fusion checkpoint functionally
        # unchanged before BRSS training.
        self.boundary_state_scale = nn.Parameter(torch.zeros(3)) if self.boundary_estimator else None
        self.boundary_residual_scale = nn.Parameter(torch.zeros(2)) if self.boundary_estimator else None
        self.last_boundary_aux = None

    def _boundary_context(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | None:
        if self.boundary_estimator is None or not self.boundary_estimator.use_global_context:
            return None
        scale = min(self.global_scales) if self.global_scales else 4
        h, w = x.shape[-2:]
        hg, wg = max(1, (h + scale - 1) // scale), max(1, (w + scale - 1) // scale)
        x_ctx = F.adaptive_avg_pool2d(x, (hg, wg)).mean(1, keepdim=True)
        y_ctx = F.adaptive_avg_pool2d(y, (hg, wg)).mean(1, keepdim=True)
        context = torch.cat([x_ctx, y_ctx], dim=1)
        return F.interpolate(context, (h, w), mode="bilinear", align_corners=False)

    @staticmethod
    def _four_direction_map(value: torch.Tensor) -> torch.Tensor:
        """Convert B1HW reliability to the four SS2D scan directions."""
        b, _, h, w = value.shape
        length = h * w
        hw = value.view(b, 1, length)
        wh = value.transpose(2, 3).contiguous().view(b, 1, length)
        forward = torch.stack([hw, wh], dim=1)
        return torch.cat([forward, torch.flip(forward, dims=[-1])], dim=1)

    def _global_map(self, x_conv, y_conv, H: int, W: int, scale: int):
        B = x_conv.shape[0]
        Hg = max(1, (H + scale - 1) // scale)
        Wg = max(1, (W + scale - 1) // scale)
        x_g = F.adaptive_avg_pool2d(x_conv, output_size=(Hg, Wg))
        y_g = F.adaptive_avg_pool2d(y_conv, output_size=(Hg, Wg))

        if self.global_op == "scan":
            x_seq_g, y_seq_g = self.forward_core(x_g, y_g)
            x_map_g = torch.transpose(x_seq_g, 1, 2).contiguous().view(B, Hg, Wg, -1).permute(0, 3, 1, 2)
            y_map_g = torch.transpose(y_seq_g, 1, 2).contiguous().view(B, Hg, Wg, -1).permute(0, 3, 1, 2)
        elif self.global_op == "pool1x1":
            # 1x1 path without extra learned parameters: keeps only pooled low-frequency context.
            x_map_g = x_g
            y_map_g = y_g
        else:  # dw3x3
            # Lightweight local substitute: depthwise 3x3 averaging on coarse grid.
            c = x_g.shape[1]
            k = x_g.new_ones((c, 1, 3, 3)) / 9.0
            x_map_g = F.conv2d(x_g, k, padding=1, groups=c)
            y_map_g = F.conv2d(y_g, k, padding=1, groups=c)

        x_map_g = F.interpolate(x_map_g, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        y_map_g = F.interpolate(y_map_g, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        return x_map_g, y_map_g

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def recover_flip(self, out_y, B, H, W, L):
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

    def forward_core(self, x: torch.Tensor, y: torch.Tensor, reliability=None):
        """
        b: batch
        l: W * H
        k: 4
        b: batch
        r: rank?
        """
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3)\
                              .contiguous().view(B, -1, L)], dim=1)\
                                .view(B, 2, -1, L)
        y_hwwh = torch.stack([y.view(B, -1, L), torch.transpose(y, dim0=2, dim1=3)\
                              .contiguous().view(B, -1, L)], dim=1)\
                                .view(B, 2, -1, L)

        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        ys = torch.cat([y_hwwh, torch.flip(y_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        x_dts, x_Bs, x_Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) 
        x_dts = torch.einsum("b k r l, k d r -> b k d l", x_dts.view(B, K, -1, L), self.dt_projs_weight)

        y_dbl = torch.einsum("b k d l, k c d -> b k c l", ys.view(B, K, -1, L), self.y_proj_weight)
        y_dts, y_Bs, y_Cs = torch.split(y_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) 
        y_dts = torch.einsum("b k r l, k d r -> b k d l", y_dts.view(B, K, -1, L), self.dt_projs_weight)
        
        mark = torch.einsum("b k C l, c C -> b k c l", torch.cat([xs, ys], dim=2), self.mark_proj)

        if reliability is not None and self.boundary_mode in {"state", "full"}:
            ir_rel = self._four_direction_map(reliability.ir)
            vi_rel = self._four_direction_map(reliability.vi)
            shared_rel = self._four_direction_map(reliability.shared)
            state_scale = torch.tanh(self.boundary_state_scale)
            ir_gain = 1.0 + state_scale[0] * (ir_rel - vi_rel)
            vi_gain = 1.0 + state_scale[0] * (vi_rel - ir_rel)
            shared_gain = 1.0 + state_scale[1] * shared_rel
            delta_ir = 1.0 + state_scale[2] * (1.0 - ir_rel)
            delta_vi = 1.0 + state_scale[2] * (1.0 - vi_rel)

            x_Bs = x_Bs * ir_gain
            x_Cs = x_Cs * ir_gain * shared_gain
            y_Bs = y_Bs * vi_gain
            y_Cs = y_Cs * vi_gain * shared_gain
            x_dts = x_dts * delta_ir
            y_dts = y_dts * delta_vi
            mark = mark * shared_gain

        xs = xs + mark
        ys = ys + mark
        
        xs = torch.cat([xs.unsqueeze(3).transpose(-1, -2), \
                        ys.unsqueeze(3).transpose(-1, -2)], dim=-1).contiguous().view(B, K, C, L * 2)
        dts = torch.cat([x_dts.unsqueeze(3).transpose(-1, -2), \
                        y_dts.unsqueeze(3).transpose(-1, -2)], dim=-1).contiguous().view(B, K, y_dts.shape[2], L * 2)
        Bs = torch.cat([x_Bs.unsqueeze(3).transpose(-1, -2), \
                        y_Bs.unsqueeze(3).transpose(-1, -2)], dim=-1).contiguous().view(B, K, y_Bs.shape[2], L * 2)
        Cs = torch.cat([x_Cs.unsqueeze(3).transpose(-1, -2), \
                        y_Cs.unsqueeze(3).transpose(-1, -2)], dim=-1).contiguous().view(B, K, y_Cs.shape[2], L * 2)

        xs = xs.float().view(B, -1, L * 2) # (b, k * d, l) 

        # context-aware params
        dts = dts.contiguous().float().view(B, -1, L * 2) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L * 2) # (b, k, d_state, l) 
        Cs = Cs.float().view(B, K, -1, L * 2) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)

        # context-independent params
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        out_y = out_y.view(B, K, -1, L * 2)

        return self.recover_flip(out_y[:, :, :, 0::2], B=B, H=H, W=W, L=L), \
            self.recover_flip(out_y[:, :, :, 1::2], B=B, H=H, W=W, L=L)

    def forward(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        yz = self.in_proj(y)
        x, z_x = xz.chunk(2, dim=-1)
        y, z_y = yz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        y = y.permute(0, 3, 1, 2).contiguous()

        x = self.act(self.conv2d(x))
        y = self.act(self.conv2d(y))
        x_conv, y_conv = x, y

        reliability = None
        if self.boundary_estimator is not None:
            reliability = self.boundary_estimator(
                x_conv,
                y_conv,
                global_context=self._boundary_context(x_conv, y_conv),
            )

        x, y = self.forward_core(x_conv, y_conv, reliability=reliability) # (B, D, L)

        x = torch.transpose(x, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        if self.global_scales:
            gate_features = torch.cat([x_conv, y_conv], dim=1)
            gates = cssc_gate_values(gate_features, self.global_gates, self.dynamic_gate, H, W)
            for idx, scale in enumerate(self.global_scales):
                x_map_g, y_map_g = self._global_map(x_conv, y_conv, H, W, scale)
                gate = gates[:, idx].view(B, 1, 1, 1)
                x = x + gate * x_map_g
                y = y + gate * y_map_g

        if reliability is not None and self.boundary_mode in {"residual", "full"}:
            residual_scale = torch.tanh(self.boundary_residual_scale)
            ir_mask = (reliability.ir + 0.5 * reliability.shared).permute(0, 2, 3, 1)
            vi_mask = (reliability.vi + 0.5 * reliability.shared).permute(0, 2, 3, 1)
            x_local = x_conv.permute(0, 2, 3, 1)
            y_local = y_conv.permute(0, 2, 3, 1)
            x = x + residual_scale[0] * ir_mask * x_local
            y = y + residual_scale[1] * vi_mask * y_local

        self.last_boundary_aux = reliability
        
        y = self.out_norm(y)
        x = self.out_norm(x)
        
        y = y * F.silu(z_y)
        x = x * F.silu(z_x)
        out_y = self.out_proj(y)
        out_x = self.out_proj(x)
        
        if self.dropout is not None:
            out_y = self.dropout(out_y)
            out_x = self.dropout(out_x)
        return out_x, out_y
