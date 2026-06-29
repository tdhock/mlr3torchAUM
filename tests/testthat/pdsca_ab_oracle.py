import torch
from libauc.losses.auc import CompositionalAUCLoss
from libauc.optimizers.pdsca import PDSCA

w = torch.zeros(1, requires_grad=True) # simulate network weights
loss_fn = CompositionalAUCLoss(margin=1.0, k=1, version='v1', device='cpu')
opt = PDSCA([w], loss_fn, lr=0.1, lr0=0.05, beta1=0.99, beta2=0.999,
            weight_decay=0.0, epoch_decay=0.0, clip_value=10.0,
            device='cpu', verbose=False) # lr0 and beta1 are for network weights

yp = torch.tensor([0.1, 0.4, 0.35, 0.8]).reshape(-1, 1)
yt = torch.tensor([0., 0., 1., 1.]).reshape(-1, 1)

opt.zero_grad()
(loss_fn(yp, yt) + 0*w.sum()).backward()
opt.step()
print("after CE :", float(loss_fn.a), float(loss_fn.b), float(loss_fn.alpha))

opt.zero_grad()
(loss_fn(yp, yt) + 0*w.sum()).backward()
opt.step()
print("after AUC:", float(loss_fn.a), float(loss_fn.b), float(loss_fn.alpha))

# uv run --with 'libauc==1.4.0' --with torch --with 'numpy<2' --python 3.11 pdsca_ab_oracle.py