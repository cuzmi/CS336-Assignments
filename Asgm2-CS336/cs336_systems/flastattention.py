import torch

import math
import torch

# NOTE: falshattention的核心实现部分 - [By: Weijie] - 2026/03/18
class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        """
        q: (B, Nq, D)
        k: (B, Nk, D)
        v: (B, Nk, D)
        is_causal: bool

        return:
            out: (B, Nq, D)
        """
        device = q.device
        B, N_q, d = q.shape
        _, N_k, d_v = v.shape
        # NOTE: 确保矩阵乘法可以用 - [By: Weijie] - 2026/03/19
        assert k.shape[0] == B and k.shape[2] == d
        assert v.shape[0] == B and v.shape[1] == N_k

        scale = d ** -0.5
        block_q = 64
        block_k = 64
            # NOTE: tiling 长度 - [By: Weijie] - 2026/03/19
        n_q_blocks = math.ceil(N_q / block_q)
        n_k_blocks = math.ceil(N_k / block_k)

        # 输出用输入 dtype，内部累积用 float32 更稳定
        out = torch.empty((B, N_q, d_v), device=device, dtype=q.dtype)
        lse = torch.empty((B, N_q), device=device, dtype=torch.float32)

        """
        里面需要维护的只有max(全局)、l_i(分母累积和)、o_i(旧v乘积更新 + 新的tile乘积)
        还是要理清楚online softmax的逻辑, 乘完v之后最后才用归一化l_i
        """
        for b in range(B): 
            Q = q[b]   # (N_q, d)
            K = k[b]   # (N_k, d)
            V = v[b]   # (N_k, d_v)

            for qi in range(n_q_blocks):
                q_start = qi * block_q
                q_end = min((qi + 1) * block_q, N_q)

                Q_i = Q[q_start:q_end, :]          # (q_blk, d)
                q_blk = q_end - q_start

                # online softmax 状态
                m_i = torch.full((q_blk,), float("-inf"), device=device, dtype=torch.float32)
                l_i = torch.zeros((q_blk,), device=device, dtype=torch.float32)
                o_i = torch.zeros((q_blk, d_v), device=device, dtype=torch.float32)

                for ki in range(n_k_blocks):
                    k_start = ki * block_k
                    k_end = min((ki + 1) * block_k, N_k)

                    K_i = K[k_start:k_end, :]      # (k_blk, d)
                    V_i = V[k_start:k_end, :]      # (k_blk, d_v)

                    # 当前 tile 的分数
                    score_i = (Q_i @ K_i.transpose(-1, -2)).to(torch.float32) * scale  # (q_blk, k_blk)

                    if is_causal:
                        q_idx = torch.arange(q_start, q_end, device=device)[:, None]
                        k_idx = torch.arange(k_start, k_end, device=device)[None, :]
                        causal_mask = q_idx >= k_idx
                        score_i = score_i.masked_fill(~causal_mask, float("-inf"))

                    # 当前 tile 的最大值
                    m_tile = torch.max(score_i, dim=1).values          # (q_blk,)
                    m_new = torch.maximum(m_i, m_tile)                 # (q_blk,)

                    # 旧状态换到新基准
                    exp_m_scale = torch.exp(m_i - m_new)               # (q_blk,)

                    # 当前 tile 在新基准下的指数项
                    tile_numerator = torch.exp(score_i - m_new[:, None])   # (q_blk, k_blk)

                    # 更新 online softmax 状态
                    l_i = l_i * exp_m_scale + torch.sum(tile_numerator, dim=1)                # (q_blk,)
                    o_i = o_i * exp_m_scale[:, None] + tile_numerator @ V_i.to(torch.float32) # (q_blk, d_v)

                    m_i = m_new

                # 最终归一化
                o_i = o_i / l_i[:, None]
                lse_i = m_i + torch.log(l_i)

                out[b, q_start:q_end] = o_i.to(q.dtype)
                lse[b, q_start:q_end] = lse_i

        # 测试会找 shape == (B, N_q) 的 saved tensor，所以 lse 必须保存
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.block_q = block_q
        ctx.block_k = block_k

        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        grad_out: (B, Nq, Dv)
        returns: dq, dk, dv, None
        """
        q, k, v, out, lse = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale
        block_q = ctx.block_q
        block_k = ctx.block_k

        device = q.device
        B, N_q, d = q.shape
        _, N_k, d_v = v.shape

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        n_q_blocks = math.ceil(N_q / block_q)
        n_k_blocks = math.ceil(N_k / block_k)

        # TODO: 这里和刚刚做的数学的softmax的求奥有什么联系吗
        #  - [By: Weijie] - 2026/03/19
        for b in range(B):
            Q = q[b]              # (Nq, D)
            K = k[b]              # (Nk, D)
            V = v[b]              # (Nk, Dv)
            O = out[b].to(torch.float32)         # (Nq, Dv)
            L = lse[b]                           # (Nq,)
            dO = grad_out[b].to(torch.float32)   # (Nq, Dv)

            # D_i = sum_j dO_ij * O_ij
            D = torch.sum(dO * O, dim=-1)  # (Nq,)

            for kj in range(n_k_blocks):
                k_start = kj * block_k
                k_end = min((kj + 1) * block_k, N_k)

                K_j = K[k_start:k_end]  # (k_blk, D)
                V_j = V[k_start:k_end]  # (k_blk, Dv)

                dK_j = torch.zeros((k_end - k_start, d), device=device, dtype=torch.float32)
                dV_j = torch.zeros((k_end - k_start, d_v), device=device, dtype=torch.float32)

                for qi in range(n_q_blocks):
                    q_start = qi * block_q
                    q_end = min((qi + 1) * block_q, N_q)

                    Q_i = Q[q_start:q_end]                       # (q_blk, D)
                    dO_i = dO[q_start:q_end]                     # (q_blk, Dv)
                    L_i = L[q_start:q_end]                       # (q_blk,)
                    D_i = D[q_start:q_end]                       # (q_blk,)

                    # 重算分数
                    S_ij = (Q_i @ K_j.transpose(0, 1)).to(torch.float32) * scale  # (q_blk, k_blk)

                    if is_causal:
                        q_idx = torch.arange(q_start, q_end, device=device)[:, None]
                        k_idx = torch.arange(k_start, k_end, device=device)[None, :]
                        causal_mask = q_idx >= k_idx
                        S_ij = S_ij.masked_fill(~causal_mask, float("-inf"))

                    # P_ij = exp(S_ij - L_i)
                    P_ij = torch.exp(S_ij - L_i[:, None])  # (q_blk, k_blk)

                    # dV += P^T @ dO
                    dV_j += P_ij.transpose(0, 1) @ dO_i

                    # dP = dO @ V^T
                    dP_ij = dO_i @ V_j.to(torch.float32).transpose(0, 1)

                    # dS = P * (dP - D)
                    dS_ij = P_ij * (dP_ij - D_i[:, None])

                    # dQ += dS @ K * scale
                    dq[b, q_start:q_end] += (dS_ij @ K_j.to(torch.float32) * scale).to(q.dtype)

                    # dK += dS^T @ Q * scale
                    dK_j += (dS_ij.transpose(0, 1) @ Q_i.to(torch.float32)) * scale

                dk[b, k_start:k_end] += dK_j.to(k.dtype)
                dv[b, k_start:k_end] += dV_j.to(v.dtype)

        return dq, dk, dv, None


def get_flashattention_autograd_function_pytorch():
    return FlashAttentionPyTorch()