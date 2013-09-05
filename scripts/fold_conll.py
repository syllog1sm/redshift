import plac
import sys
from pathlib import Path


def main(n, in_loc, out_dir):
    n = int(n)
    in_loc = Path(in_loc)
    out_dir = Path(out_dir)
    sents = in_loc.open().read().strip().split('\n\n')
    size = len(sents) / n
    for i in range(n):
        start = i * size
        if i == n - 1:
            end = len(sents)
        else:
            end = start + size
        train = sents[:start] + sents[end:]
        test = sents[start:end]
        print i, len(train), len(test)
        out_dir.join(str(i) + '.train').open('w').write(u'\n\n'.join(train))
        out_dir.join(str(i) + '.test').open('w').write(u'\n\n'.join(test))

if __name__ == '__main__':
    plac.call(main)
