import sys
import re
import plac

def main(pos_loc, mrg_loc):
    pos_sents = open(pos_loc).read().strip().split('\n')
    mrg_sents = open(mrg_loc).read().strip().split('\n')
    assert len(pos_sents) == len(mrg_sents)
    tag_re = re.compile(r'([^() ]+) (?=[^(])')
    for pos_sent, mrg_sent in zip(pos_sents, mrg_sents):
        pos_tags = [tok.rsplit(r'/', 1)[1] for tok in pos_sent.split()]
        mrg_sent = tag_re.sub('--POS-- ', mrg_sent)
        count = mrg_sent.count('--POS--')
        assert len(pos_tags) == count, '%d vs %d' % (len(pos_tags), count)
        for tag in pos_tags:
            mrg_sent = mrg_sent.replace('--POS--', tag, 1)
        print mrg_sent


if __name__ == '__main__':
    plac.call(main)



