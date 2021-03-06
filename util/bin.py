import math


class Bin:

    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.counter = 0
        self.prob = None

    def __str__(self):
        return f"[min: {self.min},max: {self.max},counter: {self.counter}, prob: {self.prob}]"

    def __repr__(self):
        return self.__str__()


def from_interval(interval):
    min = interval[0] if not interval[0] == -math.inf else -999999
    max = interval[1] if not interval[1] == math.inf else 999999
    return Bin(min, max)
