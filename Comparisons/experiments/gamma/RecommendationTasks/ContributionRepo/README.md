反正这个工作之后主要是我做了, 那我就瞎说了. 反正也没别人看. 

这个目录下有好几个数据文件. 解释一下. 假设当前目录为 ContributionRepo. 


### data/ 

这个目录下的文件都是拿来训练模型用的.

#### data/user_watch_repos.json: 

由 collect_data.py 获得. 

这个*中间文件*是用来训练模型的. 它是一个列表, 列表项为: 
    

```javascript
    [
        47071,  // 某个 contributor 的ID
        13753,  // 正例, 是该 contributor 贡献过的仓库 ID
        25366   // 负例, 该 contributor 没有贡献过的仓库 ID. 
    ]
```

正例和负例是从一个集合得到的(从集合中拿就是正例, 否则为负例): 该用户所有watch或commit过的仓库的集合.


#### data/{test.json,valid.json,train.json}

就是将上面的文件分割出来, 得到的三个临时文件. 也是用来训练数据的. 


### metric

#### metric/data/dataset_valid_test.json

这个文件由 metric/collect_data.py 取得. 

对于其中的某个元素: 

```javascript
    [ 54171, [ 18950, ..., 19967 ], [ 15830 ] ]
```

- 第一个分量是某个贡献者的 id; 
- 第二个分量是这个贡献者贡献过的所有仓库.
- 第三个分量是贡献过的其中某个仓库...是从数据来源的 pos_repository 得到的. 

整个数据是从 data/test.json 和 data/valid.json 的交得到的; 

