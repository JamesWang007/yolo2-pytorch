import time


class Barrier(object):
    def __init__(self):
        self.t_list = [time.time()]
        self.idx_list = [0]
        pass

    def add(self, idx):
        self.t_list.append(time.time())
        self.idx_list.append(idx)

    def print(self):
        sum = 0.0
        for i in range(len(self.t_list) - 1):
            diff = self.t_list[i + 1] - self.t_list[i]
            print(self.idx_list[i + 1], '{:.4f} seconds'.format(diff))
            sum += diff
        print('--- sum {:.4f}'.format(sum))
