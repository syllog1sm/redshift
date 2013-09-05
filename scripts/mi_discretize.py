"""Find the best mutual-information discretisation for a continuous variable"""

import plac
from collections import defaultdict
import math

def get_entropy(instances, cut=None):
    joint_freqs = defaultdict(int)
    single_freqs = defaultdict(int)
    total = 0
    for clas, value in instances:
        if cut is None:
            value = 0
        elif value < cut:
            value = 0
        else:
            value = 1
        joint_freqs[(clas, value)] += 1
        single_freqs[value] += 1
        total += 1

    p = 0.0
    for pair, freq in joint_freqs.items():
        if freq:
            joint_prob = float(freq) / total
            ev_prob = float(single_freqs[pair[1]]) / total
            p += (joint_prob + math.log(ev_prob / joint_prob, 2))
    return p


def main(in_loc):
    instances = []
    for line in open(in_loc):
        if not line.strip(): continue
        try:
            class_, value = line.split()
        except:
            print line
            raise
        instances.append((class_, float(value)))
    print get_entropy(instances, cut=None)
    for cut in range(20):
        cut = float(cut) / 20
        print cut, get_entropy(instances, cut=cut)


if __name__ == '__main__':
    plac.call(main)
