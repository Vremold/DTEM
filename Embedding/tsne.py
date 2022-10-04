# 将deepwalk或node2vec得到的高维嵌入向量降维成2维向量

from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import sys

sourcefile = sys.argv[1]
targetfile = sys.argv[2]
with open(sourcefile, 'r') as sor:
    tarr = []
    cnt = 0
    for line in sor:
        cnt += 1
        if cnt == 1:
            continue
        line = line[line.find(' ') + 1:]
        arre = line.split()
        tarr.append([arre])
    
X = np.array(tarr)
print(X.ndim)
embeddings = TSNE(n_jobs=4).fit_transform(X)

with open('result/' + targetfile, 'w') as wri:
    for i in embeddings:
        wri.write(str(i[0]) + ' ' + str(i[1]) + '\n')