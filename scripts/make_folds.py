import plac
from pathlib import Path
import re

def main(tb_loc, out_loc):
    tb_loc = Path(tb_loc)
    out_loc = Path(out_loc)
    sections = '02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21'.split(',')
    for test_sec in sections:
        train_secs = [s for s in sections if s != test_sec]
        files = []
        for train_sec in train_secs:
            files.extend(f for f in tb_loc.join(train_sec) if f.parts[-1].endswith('.mrg'))
        out_file = out_loc.join('not%s.mrg' % test_sec).open('w')
        for file_ in files:
            out_file.write(file_.open().read().strip())
            out_file.write(u'\n')
        out_file.write(u'\n')
        out_file.close()
        test_file = out_loc.join('%s.mrg' % test_sec).open('w')
        test_text = out_loc.join('%s.txt' % test_sec).open('w')
        sent_re = re.compile(r'^\( \(')
        for file_ in tb_loc.join(test_sec):
            test_file.write(file_.open().read().strip())
            test_file.write(u'\n')
            sentences = sent_re.split(file_.open().read().strip())
            print repr(sentences[0])
        test_file.write(u'\n')
        test_file.close()

if __name__ == '__main__':
    plac.call(main)
