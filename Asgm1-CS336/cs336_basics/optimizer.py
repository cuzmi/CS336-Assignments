"""
optimizer 属性: param_groups, state, defaults
通过外包装和利用state 来实现自定义的optimizer方法, 通过遍历param_groups对每一个参数进行自定义更新
//
1. 原地修改, 形式为 tensor.xxx_
"""
import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):

    def __init__(self, params, lr = 1e-3, betas = [0.9, 0.999], eps = 1e-8, weight_decay = 1e-2):
        # 判断内容是否合法， 参数确认
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid bates vale :{betas}")
        if eps < 0:
            raise ValueError(f"Invalid eps value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        # 创建self实例， 结果是 optimizer.(param_groups, state, defaults, ..). 其中param_groups的每个group都是param，lr等参数形式的，用户可以只提供部分参数，Optimizer 会用 defaults 自动补齐，使所有 group 结构一致 
        defaults = dict(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)
        super().__init__(params, defaults)

    def step(self, closure = None):
        # closure
        loss = None
        if closure: # closure 后的处理？
            # NOTE: closure不会更新参数，只是计算了梯度方向，更新还需要走一步step，完成下面内容 - [By: Weijie] - 2026/03/09
            with torch.enable_grad():
                loss = closure()
        
        # 定位参数 进行更新？
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad == None:
                    continue
                
                grad = p.grad
                # 更新前初始化 state 和 self.state 指向的是同一个对象
                state = self.state[p]

                # ====== step 0 state 初始化， 自定义 ====
                if len(state) == 0:
                    state['step'] = 0 

                    state['exp_av'] = torch.zeros_like(p, memory_format = torch.preserve_format)
                    state['exp_av_sq'] = torch.zeros_like(p, memory_format = torch.preserve_format)

                exp_av, exp_av_sq = state['exp_av'], state['exp_av_sq']
                state['step'] += 1
                step = state['step']

                # ======= step 1 权重衰减 ====== 
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # ======= step 2 更新一二阶矩 =======  state 记录的是不包含bias correction的统计量，因为公式中 mt 并没有修正
                exp_av = exp_av.mul_(betas[0]).add_(grad, alpha = 1 - betas[0])
                exp_av_sq = exp_av_sq.mul_(betas[1]).addcmul_(grad, grad, value = 1 - betas[1])

                # ===== step 3 修正偏差 ====== 
                bias_correction1 = 1 - betas[0] ** step
                bias_correction2 = 1 - betas[1] ** step

                # ===== step 4 计算步长并更新 ========
                step_size = lr / bias_correction1

                demon = exp_av_sq.sqrt() / (math.sqrt(bias_correction2)) + eps

                p.data.addcdiv_(exp_av, demon, value = - step_size)

        return loss
