import random
from math import exp
from typing import List


def softmax(values: List[float]) -> List[float]:
    exp_values = [exp(x) for x in values]
    sum_exp_values = sum(exp_values)
    norm_values = [y / sum_exp_values for y in exp_values]
    return norm_values

def main():
    data = [random.uniform(-10.0, 10.0) for i in range(100)]
    softmax_data = softmax(data)
    print(softmax_data)

main()