import torch
from libauc.losses.auc import AUCMLoss
s = torch.tensor([0.1, 0.4, 0.35, 0.8]).view(-1, 1)
y = torch.tensor([0., 0., 1., 1.]).view(-1, 1)
def golden(a, b, al, m=1.0, version='v1'):
    f = AUCMLoss(margin=m, version=version)
    with torch.no_grad():
        f.a.fill_(a); f.b.fill_(b); f.alpha.fill_(al)
    out = float(f(s, y))
    return out / (0.5 * 0.5) if version == 'v1' else out
print("v2 a=0   b=0   al=0  :", golden(0, 0, 0))                      # 0.4662500321865082
print("v2 a=0.3 b=0.6 al=0.5:", golden(0.3, 0.6, 0.5))                # 0.6962500214576721
print("v1 a=0   b=0   al=0  :", float(AUCMLoss(margin=1.0, version='v1')(s, y)))
                                                                      # 0.11656250804662704
print("LibAUC v2 (buggy)    :", golden(0.3, 0.6, 0.5, version='v2'))  # 0.5712500214576721