import torch    # 如正常则静默
a = torch.Tensor([1.])    # 如正常则静默
a.cuda()    # 如正常则返回"tensor([ 1.], device='cuda:0')"
from torch.backends import cudnn # 如正常则静默
print(cudnn.is_acceptable(a.cuda()))    # 如正常则返回 "True"