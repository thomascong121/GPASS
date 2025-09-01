import numpy as np

data_root = '/root/autodl-tmp/CAGAN/results/tt/normalised/'
center = ['HD', 'TT', 'YC', 'ZD', 'ZJ']
base_all = []
pred_all = []

for c in center:
    center_root = f'{data_root}/{c}'
    base_all.append(np.load(f'{center_root}/nim_base.npy'))
    pred_all.append(np.load(f'{center_root}/nim_pred.npy'))

base_all = np.concatenate(base_all)
pred_all = np.concatenate(pred_all)
mean = np.mean(base_all)
# sample std (ddof=1) is common for reporting; use ddof=0 for population std
std_sample = np.std(base_all, ddof=1)
cov_sample = std_sample / (mean + 1e-12)
print('Base: %.3f, %.3f, %.3f' % (mean, std_sample, cov_sample))

mean = np.mean(pred_all)
std_sample = np.std(pred_all, ddof=1)
cov_sample = std_sample / (mean + 1e-12)
print('Pred: %.3f, %.3f, %.3f' % (mean, std_sample, cov_sample))


