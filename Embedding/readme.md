## Deepwalk 图嵌入
- 下载[Deepwalk](https://github.com/phanein/deepwalk)并安装
- 执行 `deepwalk --input graph/INPUT --output embedded/OUTPUT` (此处INPUT为GraphBuilding得到的adjlist或edgelist)

## Node2vec 图嵌入
- 下载[SNAP](https://github.com/snap-stanford/snap)，进入snap-master/examples/node2vec文件夹，使用`make all`进行编译
- 执行 `./node2vec -i:graph/INPUT -o:emb/OUTPUT` (此处INPUT为GraphBuilding得到的adjlist或edgelist)

## 嵌入可视化
- `pip install MulticoreTSNE`
- 执行 `python tsne.py INPUT OUTPUT` (此处INPUT为Deepwalk或Node2vec输出的嵌入)得到2维向量
- 执行 `python visual.py EMB_INPUT 2D_INPUT OUTPUT` (此处EMB_INPUT为deepwalk或node2vec输出的高维向量，2D_INPUT为tsne输出的2维向量)得到gexf文件，可以使用Gephi软件得到图像