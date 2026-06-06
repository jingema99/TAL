# cifar100_tal.py
import torch
import torch.nn as nn



# -------- TAL (batched, frequency-aligned alpha) --------

class TAL_Loss(nn.Module):
    """
    TAL with uniform-normalized negative supervision in Q-update:
      Q_i^+ = t * ( Q_i + (Np(i)/N) - (Nn(i)/N) * s_i(Q) / (C-1) )
    and frequency-aligned alpha consistent with that update.

    neg_mode:
      - "normalized": use 1/(C-1) in Q-update (default in this patch)
      - "vanilla":    original (no normalization) — kept for ablation
    """

    def __init__(self, t: float = 0.99, r: float = 3.0, eps: float = 1e-12,
                 neg_mode: str = "normalized", recalib_alpha: bool = True):
        super().__init__()
        assert 0.0 < t < 1.0 and r > 0.0
        assert neg_mode in ("normalized", "vanilla")
        self.t, self.r, self.eps = float(t), float(r), float(eps)
        self.neg_mode = neg_mode
        self.recalib_alpha = bool(recalib_alpha)

        self.Qmax = self.t / (1.0 - self.t)
        self.num_classes = None
        self.alpha_loss = None
        self.register_buffer("Q", torch.tensor([], dtype=torch.float32))

    # ---- root solver for kappa * x^r + x - p = 0 (unique root in (0,1)) ----
    @staticmethod
    def _solve_x_star_kappa(p: float, r: float, kappa: float, tol: float = 1e-12) -> float:
        if r == 1.0:
            return p / (1.0 + kappa)
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            F = kappa * (mid ** r) + mid - p
            if F > 0: hi = mid
            else: lo = mid
            if hi - lo < tol: break
        return 0.5 * (lo + hi)

    @staticmethod
    def calibrate_alpha_freq(num_classes: int, r: float, neg_mode: str) -> float:
        C = float(num_classes)
        p = 1.0 / C
        if neg_mode == "vanilla":
            gamma_neg = 1.0
        elif neg_mode == "normalized":
            gamma_neg = 1.0 / max(C - 1.0, 1.0)
        else:
            raise ValueError(f"Unknown neg_mode: {neg_mode}")

        # steady-state equation: kappa * x^r + x - p = 0
        kappa = (1.0 - p) * gamma_neg
        x_star = TAL_Loss._solve_x_star_kappa(p, r, kappa)
        s_star = x_star ** r
        alpha = float(1.0 / s_star)
        print("alpha = ", alpha)
        return alpha # enforce alpha * s* = 1

    @torch.no_grad()
    def update_class_num(self, num_classes: int):
        assert num_classes >= 1
        oldC = int(self.Q.numel())
        newC = int(num_classes)
        if oldC == 0:
            self.Q = torch.zeros(newC, dtype=torch.float32)
        else:
            assert newC > oldC, "expect strictly increasing class count"
            newQ = torch.zeros(newC, dtype=self.Q.dtype, device=self.Q.device)
            newQ[:oldC] = self.Q
            self.Q = newQ
        self.num_classes = newC
        if self.alpha_loss is None or self.recalib_alpha:
            self.alpha_loss = self.calibrate_alpha_freq(newC, self.r, self.neg_mode)

    @torch.no_grad()
    def group_mean_Q_s(self, group_size: int = 10, upto_classes: int = None):
        """
        Return per-group mean of:
          - q_mean: mean(Q)
          - w_mean: mean(w)=mean(alpha*s) (actual weight used in loss, no floor in backup version)
        """
        if group_size <= 0:
            raise ValueError("group_size must be > 0")
        if self.num_classes is None or self.Q.numel() == 0:
            return []

        C = int(self.num_classes) if upto_classes is None else int(upto_classes)
        C = max(0, min(C, int(self.Q.numel())))
        if C == 0:
            return []

        Q = self.Q[:C].detach().float().cpu()
        Qmax = float(self.Qmax)
        r = float(self.r)
        alpha = float(self.alpha_loss if self.alpha_loss is not None else 1.0)

        s = (Q / Qmax).clamp_min(0.0).pow(r)
        w = alpha * s

        stats = []
        for start in range(0, C, group_size):
            end = min(start + group_size, C)
            stats.append({
                "range": f"{start}-{end - 1}",
                "q_mean": float(Q[start:end].mean().item()),
                "w_mean": float(w[start:end].mean().item()),
            })
        return stats

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        N, C = logits.shape
        assert self.num_classes == C and self.alpha_loss is not None

        # ensure Q dtype/device
        if self.Q.device != logits.device or self.Q.dtype != logits.dtype:
            self.Q = self.Q.to(device=logits.device, dtype=logits.dtype)

        t, r, Qmax = self.t, self.r, self.Qmax
        dev, dtype = logits.device, logits.dtype

        # ---- loss: only negatives reweighted by alpha * s(Q) (same as before) ----
        Q_pre = self.Q
        alpha = torch.tensor(self.alpha_loss, dtype=dtype, device=dev)
        s = (Q_pre / Qmax).clamp_min(0).pow(r)                    # [C]
        eps_t = torch.tensor(self.eps, dtype=dtype, device=dev)

        log_alpha_s = torch.log((alpha * s).clamp_min(eps_t)).to(dtype)  # [C]

        adjusted = logits + log_alpha_s                                   # [N,C]
        true_col = targets.view(-1, 1)
        true_vals = logits.gather(1, true_col)
        adjusted = adjusted.scatter(1, true_col, true_vals)

        lse = torch.logsumexp(adjusted, dim=1)
        loss = (lse - true_vals.squeeze(1)).mean()

        # ---- Q update (UNIFORM-NORMALIZED NEGATIVE) ----
        with torch.no_grad():
            Np = torch.bincount(targets, minlength=C).to(dtype)  # [C]
            Nn = float(N) - Np
            if self.neg_mode == "normalized":
                denom = max(C - 1.0, 1.0)
                Q_post = t * (Q_pre + (Np / float(N)) - (Nn / float(N)) * (s / denom))
            else:  # vanilla (original)
                Q_post = t * (Q_pre + (Np / float(N)) - (Nn / float(N)) * s)
            self.Q = Q_post


        return loss
