import torch

# Parameters for training and modeling
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
lr = (1e-6, 1e-3)
ckpt_dir = './bayes'

# Parameters for Bayesian Optimization
init_points = 5
n_iter = 20
