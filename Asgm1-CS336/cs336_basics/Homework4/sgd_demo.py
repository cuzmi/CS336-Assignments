import torch

from torch.optim import SGD

for lr in [1e1, 1e2, 1e3]:
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    
    opt = SGD([weights], lr=lr)

    start_loss = end_loss = 1e5

    for t in range(100):

        opt.zero_grad() # Reset the gradients for all learnable parameters. 
        loss = (weights**2).mean() # Compute a scalar loss value. 
        start_loss = loss if t == 0 else start_loss
        end_loss = loss if t == 99 else end_loss
        if t % 10 == 0:
            print(loss.cpu().item()) 
        loss.backward() # Run backward pass, which computes gradients. 
        opt.step() # Run optimizer step.

    print(f"After 100 iteration with lr:{lr}, our loss decrease:{start_loss - end_loss:.3f}")