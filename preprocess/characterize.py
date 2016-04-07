# characterize.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import sys
import codecs

if __name__ == '__main__':
    fd = codecs.open(sys.argv[1], 'r', 'utf-8')
    fw = codecs.open(sys.argv[2], 'w', 'utf-8')

    for line in fd:
        line = line.strip()
        wlist = line.split()
        clist = []

        for word in wlist:
            for char in word:
                clist.append(char)

        fw.write(' '.join(clist) + '\n')

    fd.close()
    fw.close()
