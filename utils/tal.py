import torch
import torch.nn as nn


class TAL_Loss(nn.Module):
    """
    Temporal-Adjusted Loss (TAL).

    Paper notation:
      - lambda_ is the temporal memory parameter.
      - Q tracks the temporal positive supervision strength.
      - Q_max = lambda_ / (1 - lambda_) is the upper bound of Q.
      - w(Q) = (Q / Q_max)^r rescales negative supervision.
      - alpha is the frequency alignment parameter.

    neg_mode:
      - "normalized": use 1/(C-1) in Q-update.
      - "vanilla": original update without normalization, kept for ablation.
    """

    def __init__(
        self,
        lambda_: float = None,
        r: float = 3.0,
        eps: float = 1e-12,
        neg_mode: str = "normalized",
        alpha_recalibration: bool = True,
        t: float = None,
        recalib_alpha: bool = None,
    ):
        super().__init__()
        # Backward-compatible aliases used by earlier experiment configs.
        if lambda_ is None:
            lambda_ = 0.99 if t is None else t
        if recalib_alpha is not None:
            alpha_recalibration = recalib_alpha

        assert 0.0 < lambda_ < 1.0 and r > 0.0
        assert neg_mode in ("normalized", "vanilla")
        self.lambda_ = float(lambda_)
        self.r = float(r)
        self.eps = float(eps)
        self.neg_mode = neg_mode
        self.alpha_recalibration = bool(alpha_recalibration)

        self.Q_max = self.lambda_ / (1.0 - self.lambda_)
        self.num_classes = None
        self.alpha = None
        self.register_buffer("Q", torch.tensor([], dtype=torch.float32))

        # Legacy attribute names kept for old checkpoints/debug code.
        self.t = self.lambda_
        self.Qmax = self.Q_max
        self.alpha_loss = self.alpha
        self.recalib_alpha = self.alpha_recalibration

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
    def calibrate_alpha(num_classes: int, r: float, neg_mode: str) -> float:
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
        w_star = x_star ** r
        alpha = float(1.0 / w_star)
        print("alpha = ", alpha)
        return alpha  # enforce alpha * w(Q*) = 1

    @staticmethod
    def calibrate_alpha_freq(num_classes: int, r: float, neg_mode: str) -> float:
        return TAL_Loss.calibrate_alpha(num_classes, r, neg_mode)

    @torch.no_grad()
    def update_class_num(self, num_classes: int):
        assert num_classes >= 1
        old_num_classes = int(self.Q.numel())
        new_num_classes = int(num_classes)
        if old_num_classes == 0:
            self.Q = torch.zeros(new_num_classes, dtype=torch.float32)
        else:
            assert new_num_classes > old_num_classes, "expect strictly increasing class count"
            Q_expanded = torch.zeros(new_num_classes, dtype=self.Q.dtype, device=self.Q.device)
            Q_expanded[:old_num_classes] = self.Q
            self.Q = Q_expanded
        self.num_classes = new_num_classes
        if self.alpha is None or self.alpha_recalibration:
            self.alpha = self.calibrate_alpha(new_num_classes, self.r, self.neg_mode)
            self.alpha_loss = self.alpha

    @torch.no_grad()
    def group_mean_Q_w(self, group_size: int = 10, upto_classes: int = None):
        """
        Return per-group mean of:
          - q_mean: mean(Q)
          - w_mean: mean(alpha * w(Q)), the actual negative-supervision weight.
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
        Q_max = float(self.Q_max)
        r = float(self.r)
        alpha = float(self.alpha if self.alpha is not None else 1.0)

        w_Q = (Q / Q_max).clamp_min(0.0).pow(r)
        negative_weight = alpha * w_Q

        stats = []
        for start in range(0, C, group_size):
            end = min(start + group_size, C)
            stats.append({
                "range": f"{start}-{end - 1}",
                "q_mean": float(Q[start:end].mean().item()),
                "w_mean": float(negative_weight[start:end].mean().item()),
            })
        return stats

    @torch.no_grad()
    def group_mean_Q_s(self, group_size: int = 10, upto_classes: int = None):
        return self.group_mean_Q_w(group_size=group_size, upto_classes=upto_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        N, C = logits.shape
        assert self.num_classes == C and self.alpha is not None

        # ensure Q dtype/device
        if self.Q.device != logits.device or self.Q.dtype != logits.dtype:
            self.Q = self.Q.to(device=logits.device, dtype=logits.dtype)

        lambda_, r, Q_max = self.lambda_, self.r, self.Q_max
        dev, dtype = logits.device, logits.dtype

        # ---- loss: only negatives reweighted by alpha * w(Q) ----
        Q_N = self.Q
        alpha = torch.tensor(self.alpha, dtype=dtype, device=dev)
        w_Q = (Q_N / Q_max).clamp_min(0).pow(r)                    # [C]
        eps_t = torch.tensor(self.eps, dtype=dtype, device=dev)

        log_alpha_w = torch.log((alpha * w_Q).clamp_min(eps_t)).to(dtype)  # [C]

        adjusted = logits + log_alpha_w                                    # [N,C]
        true_col = targets.view(-1, 1)
        true_vals = logits.gather(1, true_col)
        adjusted = adjusted.scatter(1, true_col, true_vals)

        lse = torch.logsumexp(adjusted, dim=1)
        loss = (lse - true_vals.squeeze(1)).mean()

        # ---- Q update (UNIFORM-NORMALIZED NEGATIVE) ----
        with torch.no_grad():
            positive_count = torch.bincount(targets, minlength=C).to(dtype)  # [C]
            negative_count = float(N) - positive_count
            if self.neg_mode == "normalized":
                denom = max(C - 1.0, 1.0)
                Q_next = lambda_ * (
                    Q_N
                    + (positive_count / float(N))
                    - (negative_count / float(N)) * (w_Q / denom)
                )
            else:  # vanilla (original)
                Q_next = lambda_ * (
                    Q_N
                    + (positive_count / float(N))
                    - (negative_count / float(N)) * w_Q
                )
            self.Q = Q_next

        return loss
