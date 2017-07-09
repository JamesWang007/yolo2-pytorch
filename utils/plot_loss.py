import matplotlib.pyplot as plt
import numpy as np

exp1 = 'kitti_baseline_v3_yf'
exp2 = 'kitti_baseline_v3'
log_file1 = '/home/cory/project/yolo2-pytorch/models/training/' + exp1 + '/train.log'  # red
log_file2 = '/home/cory/project/yolo2-pytorch/models/training/' + exp2 + '/train.log'  # blue
log1 = np.genfromtxt(log_file1, delimiter=', ')
log2 = np.genfromtxt(log_file2, delimiter=', ')


def moving_avg(x, N):
    return np.convolve(x, np.ones((N,))/N, mode='valid')

begin_index = min(0, log1.shape[0], log2.shape[0])
end_index = min(log1.shape[0], log2.shape[0])
N_avg = 5
N_log_per_epoch = 55
x = np.arange(begin_index, end_index - N_avg + 1, dtype=np.float32)
x /= N_log_per_epoch
print()
s1 = moving_avg(log1[begin_index:end_index, 2], N_avg)
s2 = moving_avg(log2[begin_index:end_index, 2], N_avg)

log_scale = True
if log_scale:
    s1 = np.log(s1)
    s2 = np.log(s2)

if log_file1 != log_file2:
    plt.plot(x, s1, 'r-', x, s2, 'b-')
else:
    plt.plot(x, s1, 'r-')

axes = plt.gca()
# plt.ylim([0, 1])
plt.show()
