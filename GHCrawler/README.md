# 介绍

## 数据介绍

1. ghs.csv文件来自于论文《jampling Projects in GitHub for MSR Studies》开源的[GitHub数据集](https://seart-ghs.si.usi.ch)上使用在**2022年之后出现更新**并且**star数目大于等于15**（统计时间截止2022年5月20日）的仓库名称。采样的数据集存放于GHCrawler/ghs.csv


## 来自周凯修的补充

好了好了, 剩下的都是我来写了. 

此目录下的所有以 crawl 开头的文件, 都是用来爬取数据的...这不是显然的么... config.py 和 path.py 两个文件是为这些爬取数据的脚本做支撑的. 

这些脚本的数据来源是 ./export/selected_repos.json, 输出到 ./raw_data/ 下的各种文件. 

然后, clean.py 会将这些文件再读一下, 再从 ./export/selected_repos.json 中筛选一下, 并做另外的一些筛选工作. 最后输出到 ./cleaned 目录. 

所以...故事很奇怪.
1. ./rawdata 下并没有这些文件; 
2. ./export 下有很多没有涉及到的文件; 
3. ./rawdata/ghs.csv 从来没有被使用过. 

猜测应该是这样的: 

1. 原本的代码中, 通过 ghs.csv, 事先筛选了50k个仓库, 存放到 ./export/selected_repos.json 中; 
2. crawl_* 等文件中, 是将这50k个仓库读一遍, 从网上爬, 爬取的结果放到 rawdata/ 中. 
3. clean.py 将这些 rawdata/ 下的文件清洗一下, 输出到 ./cleaned 目录中; 
4. 不知道怎么搞得, 总之将 rawdata/ 下的文件(除了ghs.csv)都移动到了 export/ 下.

那么, 其他地方在看代码的时候, 只要看 ./clean 下的文件就可以了. 

值得注意的文件: 

- crawl_repo_code.py 是比较值得注意的文件, 和其他几个逻辑不太一样. 
- crawl_repo_info.py 也要看看. repo_statistics.txt

@see also:
- GHCrawler/export/README.md  这个文件介绍了每个文件的数据格式. 
- README.md  介绍了目录的功能(虽然没有完全解释清楚). 
