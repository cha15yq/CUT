import os


class SaveHandler(object):
    def __init__(self, num):
        self.max_num = num
        self.save_list = []

    def append(self, path):
        if len(self.save_list) < self.max_num:
            self.save_list.append(path)
        else:
            remove_path = self.save_list[0]
            del self.save_list[0]
            self.save_list.append(path)
            if os.path.exists(remove_path):
                os.remove(remove_path)


class AverageMeter(object):
    def __init__(self):
        self.setup()

    def setup(self):
        self.value = 0
        self.total = 0
        self.count = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.count += n
        self.total += value * n
        self.avg = self.total / self.count

    def getAvg(self):
        return self.avg

    def getCount(self):
        return self.count
