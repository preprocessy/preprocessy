import math


def max(arr):
    if len(arr) == 0:
        return 0
    maxe = arr[0]
    for ele in arr:
        if ele > maxe:
            maxe = ele
    return maxe


def min(arr):
    if len(arr) == 0:
        return 0
    mine = arr[0]
    for ele in arr:
        if ele < mine:
            mine = ele
    return mine


def mean(arr):
    sum = 0
    for ele in arr:
        sum += ele
    return sum / len(arr)


def stdev(arr):
    meanv = mean(arr)
    sum_of_square_diffs = 0
    for ele in arr:
        sum_of_square_diffs += (ele - meanv) ** 2
    sum_of_square_diffs /= len(arr)
    return math.sqrt(sum_of_square_diffs)
