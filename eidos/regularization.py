import torch
import torch.nn.functional as F
"""
def sigreg_strong_loss(x, sketch_dim=64):
    
    Forces ECF(x) ~ ECF(Gaussian).
    Matches ALL Moments (Maximum Entropy Cloud).
    Exact implementation of LeJEPA Algorithm 1.
    N, C = x.size()

    # 1. Projection (The Observer)
    # Project channels down to sketch_dim
    A = torch.randn(C, sketch_dim, device=x.device)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)

    # 2. Integration Points
    t = torch.linspace(-5, 5, 17, device=x.device)

    # 3. Theoretical Gaussian CF
    exp_f = torch.exp(-0.5 * t**2)

    # 4. Empirical CF
    # proj: [N, sketch_dim]
    proj = x @ A

    # args: [N, sketch_dim, T]
    args = proj.unsqueeze(2) * t.view(1, 1, -1)

    # ecf: [sketch_dim, T] (Mean over batch)
    ecf = torch.exp(1j * args).mean(dim=0)

    # 5. Weighted L2 Distance
    # |ecf - gauss|^2 * gauss_weight
    diff_sq = (ecf - exp_f.unsqueeze(0)).abs().square()
    err = diff_sq * exp_f.unsqueeze(0)

    # 6. Integrate
    loss = torch.trapz(err, t, dim=1) * N

    return loss.mean()

def sigreg_weak_loss(x, sketch_dim=64):
    Forces Covariance(x) ~ Identity.
    Matches the 2nd Moment (Spherical Cloud).
    Taken from: https://github.com/kreasof-ai/sigreg
    
    N, C = x.size()
    # 1. Sketching (Optional for C=512, but good for consistency)
    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device) / (C ** 0.5)
        x = x @ S.T  # [N, sketch_dim]
    else:
        sketch_dim = C

    # 2. Centering & Covariance
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)

    # 3. Target Identity
    target = torch.eye(sketch_dim, device=x.device)

    # 4. Off-diagonal suppression + Diagonal maintenance
    return torch.norm(cov - target, p='fro')

"""


class SIGReg(torch.nn.Module):
    def __init__(self, sketch_dim=64):
        super().__init__()
        self.sketch_dim = sketch_dim

    def forward(self, x):
        """
        x: [B, N, D]
        """

        B, N, D = x.shape
        device = x.device

        # projection matrix (fixed per forward pass)
        A = torch.randn(D, self.sketch_dim, device=device)
        A = A / (A.norm(dim=0, keepdim=True) + 1e-6)

        t = torch.linspace(-5, 5, 17, device=device)
        exp_f = torch.exp(-0.5 * t**2)

        total_loss = 0.0

        for b in range(B):

            xb = x[b]  # [N, D]

            proj = xb @ A  # [N, sketch_dim]

            args = proj.unsqueeze(-1) * t.view(1, 1, -1)  # [N, S, T]

            ecf = torch.exp(1j * args).mean(dim=0)  # [S, T]

            diff_sq = (ecf - exp_f.unsqueeze(0)).abs() ** 2

            err = diff_sq * exp_f.unsqueeze(0)

            loss = torch.trapz(err, t, dim=1).mean()

            total_loss += loss

        return total_loss / B


class GramReg(torch.nn.Module):
    def __init__(self, temperature=0.1, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    # -------------------------------------------------
    # 1. GRAM MATRIX
    # -------------------------------------------------
    def compute_gram(self, x):
        """
        x: [B, N, D]
        returns: [B, N, N]
        """

        x = x / (x.norm(dim=-1, keepdim=True) + self.eps)

        gram = torch.matmul(x, x.transpose(-1, -2))

        return gram
    
    def local_similarity_loss(self, gram, neighbors):
        """
        gram: [B, N, N]
        neighbors: list of length B, each is List[List[int]]
        """

        B, N, _ = gram.shape
        loss = 0.0

        for b in range(B):
            G = gram[b]  # [N, N]

            for i in range(N):
                ni = neighbors[b][i]

                if len(ni) == 0:
                    continue

                # similarity profile of patch i
                gi = G[i]  # [N]

                # enforce similarity with neighbors
                gj = G[ni]  # [K, N]

                loss += F.mse_loss(
                    gi.unsqueeze(0).expand_as(gj),
                    gj
                )

        return loss / B
    
    def boundary_object_loss(self, gram, neighbors, threshold=0.5):
        """
        encourages separation of non-neighbors
        """

        B, N, _ = gram.shape
        loss = 0.0

        for b in range(B):
            G = gram[b]

            for i in range(N):

                mask = torch.ones(N, device=G.device, dtype=torch.bool)
                mask[i] = False

                # remove neighbors from "non-object region"
                mask[neighbors[b][i]] = False

                non_neighbors = G[i][mask]

                # penalize high similarity to non-neighbors
                loss += (non_neighbors ** 2).mean()

        return loss / B
    
    def forward(self, x, neighbors):
        """
        x: [B, N, D]
        neighbors: list of lists
        """

        gram = self.compute_gram(x)

        # move neighbors to init of gram -> precomputed
        local_loss = self.local_similarity_loss(gram, neighbors)

        boundary_loss = self.boundary_object_loss(gram, neighbors)

        total = local_loss + boundary_loss

        return {
            "gram_matrix": gram,
            "local_similarity_loss": local_loss,
            "boundary_loss": boundary_loss,
            "total_loss": total
        }