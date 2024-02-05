#!/usr/bin/env python3

'''
这个脚本应该会运行挺长时间的了. 

我后来发现了获取API数据的方式. 

关键词: 1829584
数据在 'GHCrawler/cleaned/repo_pr_commits.txt'中是有的, 
但这个文件太大了...而且包含了 1.83M 行的数据, 一共有将近40GB! 

我们希望, 根据这个文件, 提取到每个API的嵌入表达, 
然后将这些嵌入的平均值, 作为开发者的 API 嵌入. 

考虑到任务性质的不同, 在这个脚本中, 我们不做嵌入向量的生成, 
只希望将API分配给每个ID的开发者. 

不过即使是这样, 工作量也还是太大了. 我估计分析数据就得花上半周时间. 

'''

from mimetypes import init
from typing import Any, Dict, Generator, List, Literal, Optional, Set, Tuple
from Comparisons.experiments.general import \
    load_contributor_index, \
    load_yaml_cfg
from tqdm import tqdm
import json, pickle, re, yaml, sys, requests, time

from rich import print as rprint
from gensim.models.doc2vec import Doc2Vec

import numpy as np 

TARGET_LANG = {
    'Python','Jupyter Notebook','Go', 'Java', 
    'Scala','Kotlin','Dart','C',
    'C++','Objective-C','C#','Julia',
    'Ruby','Perl','Rust','R'
}  # len(*) = 16

cfg = load_yaml_cfg()

class ApiTextExtractor: 

    ext2lang:           Dict[str, str]  # e.g. ext['java'] = 'Java'; ext.values() == TARGET_LANG
    api_model:          Doc2Vec
    issue_name2idx:     Dict[str, int]
    idx_issue2contr:    Dict[int, int]

    def __init__(self): 
        
        # This file: cfg['general']['filepath']['lang_file']
        #   comes from: 
        #       https://github.com/github-linguist/linguist
        #           -> lib/linguist/languages.yml
        # 包含github判断语言的规则.
        with open(cfg['general']['filepath']['linguist_file']) as fp: 
            linguist = yaml.load(fp, Loader=yaml.FullLoader)
        exts = {lang: linguist[lang]['extensions'] for lang in TARGET_LANG}
        self.ext2lang = {ext[1:]: lang for lang, exts in exts.items() for ext in exts}

        # load two files: 
        # 1. The **propose issue** relationship; 
        # 2. issue name to index 
        # Needed by method `issue_name2contributor_idx`
        with open(cfg['general']['filepath']['issue_idx_file']) as fp: 
            self.issue_name2idx = json.load(fp)
        with open(cfg['general']['filepath']['contributor_propose_issue_file']) as fp:
            tbl = {}
            for line in fp:
                contr_idx, issue_idx, _ = line.split('\t')
                tbl[int(issue_idx)] = contr_idx
            self.idx_issue2contr = tbl

        # api_model: lazy load
        # @see also: `_get_api_model`
        self.api_model: Doc2Vec = None


    def issue_name2contributor_idx(self, issue_name: str) -> Optional[int]: 
        issue_name2idx:  Dict[str, int] = self.issue_name2idx
        idx_issue2contr: Dict[int, int] = self.idx_issue2contr


        issue_idx = issue_name2idx.get(issue_name)
        contr_idx = idx_issue2contr.get(issue_idx)
        # print(f'issue_name = {issue_name}')
        # print(f'issue_idx = {issue_idx}')
        # print(f'contr_idx = {contr_idx}')
        return contr_idx

    '''
        注意, 如果不设置 use_mode = False, 
        则第一次调用这个函数会很花时间, 
        因为它需要加载一个比较大的模型文件. 
    '''
    def extract_api(self, filename: str, content: str, use_model=True) -> List[str]:  
        ext = filename.split('.')[-1]
        lang = self.ext2lang.get(ext)
        if lang is None: 
            return []

        # copied from (modified): 
        # https://github.com/ExpertiseModel/EmbeddingVectors 
        #       -> git_Collect_APIs.ipynb

        apis: List[str] = []
        if lang in {
                'Python','Jupyter Notebook','Go', 'Java', 'Scala','Kotlin','Dart'
        }: apis += re.findall("(?:from|import)\s+\w*.+;*", content)
        if lang in {'C','C++','Objective-C'}:
            apis += re.findall("(?:include)\s+\w*.+,*", content)
        if lang in {'C#','Julia'}:
            apis += re.findall("(?:using)\s+\w*.+;*", content)            
        if lang in {'Ruby','Perl'}:
            apis += re.findall("(?:require)\s+\w*.+;*", content)
        if lang in {'Rust','Perl'}:
            apis += re.findall("(?:use|use crate::)\s+\w*.+;*", content)
        if lang in {'R'}: 
            apis += re.findall("(?:library)\s*\(*\s*\w*.+;*", content)


        STOP_WORDS = {'', ' ',None,'using','include','from','import','Import','From','require', 'use','crate','library','var','\n','new',
              'return','*',"'*'",'\r'} 
        
        gen: List[str] = (token for line in apis for token in re.split('( )|\/|<|>|;|,|\(|\)|\n|\r', line))
        gen: List[str] = (it for it in gen if it not in STOP_WORDS and len(it) > 2)

        if use_model: 
            gen: List[str] = (it for it in gen if self.is_in_model(it))

        return list(gen)
    
    def is_in_model(self, api: str) -> bool: 
        model = self._get_api_model()
        return api in model.wv.vocab
    
    def api2vector(self, api: str) -> np.ndarray:  # size = 200
        model = self._get_api_model()
        vec = model.wv.get_vector(api)
        return np.array(vec)  


    def is_target_lang(self, filename: str) -> bool:
        ext = filename.split('.')[-1]
        return ext in self.ext2lang        

    def _get_api_model(self) -> Doc2Vec: 
        if self.api_model is None: 
            print('loading model...', file=sys.stderr)
            self.api_model = Doc2Vec.load(cfg['alpha']['model']['dev2vec_api_file_path'])
        return self.api_model



class ApiTextCollector: 

    def __init__(self, filepath: str, count: int=-1): 
        self.count = count
        self.reader = ApiTextCollector.repo_pr_commits_reader(filepath, max_count=count)

    def extract(self): 
        klee = ApiTextExtractor()

        ret: Dict[int, Set[str]] = {} # contr_idx => apis

        total, exists, extracted = 0, 0, 0
        err = 0

        for issue_name, commit_list in tqdm(self.reader, total=self.count):
            total += 1
            contr_idx = klee.issue_name2contributor_idx(issue_name)
            if contr_idx is None: continue
            exists += 1

            flag = False
            for commit in commit_list: 
                try: 
                    filename, content = commit['filename'], commit.get('file')
                    if not klee.is_target_lang(filename): continue

                    if content is None: 
                        content = self._get_file_content(commit['raw_url'])

                    apis = klee.extract_api(filename, content)
                    if apis == []: continue
                    ret.setdefault(contr_idx, set()).update(apis)
                    flag = True
                except Exception as e: 
                    json.dump(commit, open('.error.json', 'w'))
                    err += 1
                    print(f'error count = {err}')
                    raise e
            if flag: extracted += 1

        print(f'total = {total}, exists = {exists}, extracted = {extracted}')

        with open(cfg['alpha']['raw']['user_api_file_path'], 'w') as fp: 
            for contr_idx, apis in ret.items(): 
                fp.write(f'{contr_idx}\t{json.dumps(list(apis))}\n')

    def _get_file_content(self, url: str) -> str: 
        # TODO 固然简单, 但...? 
        resp = requests.get(url)
        print(time.time())
        return resp.text


    @staticmethod
    def repo_pr_commits_reader(filepath: str, max_count=-1) -> Generator[Tuple[str, List[Any]], None, None]: 
        with open(filepath) as fp: 
            gen = (it for it, _ in zip(fp, range(max_count))) if max_count != -1 \
                else (it for it in fp)
            for line in gen: 
                # @see also C/e/config.yaml -> alpha.collect_data.api.src.repo_pr_commits_file
                repo_name, issue_idx, _, content = line.split('\t')

                issue_name = f'{repo_name}#{issue_idx}'
                content = json.loads(content)

                yield issue_name, content


def main(): 
    data_in = pickle.load(open('Comparisons/data/alpha/user_apis.pkl', 'rb'))
    klee = ApiTextExtractor()

    data_out = {
        user_id: [it for it in value if klee.is_in_model(it)]
        for user_id, value in tqdm(data_in.items())
    }

    pickle.dump(data_out, open('Comparisons/data/alpha/user_apis_filtered.pkl', 'wb'))


    # filepath = cfg['general']['filepath']['repo_pr_commits_file']
    # count = 1829584 + 10

    # klee = ApiTextCollector(filepath, count)
    # klee.extract()

def contributor_api_to_embedding(): 
    # step 1. read user_apis.pkl
    data_in  = pickle.load(open(cfg['alpha']['raw']['user_api_file_path_2'], 'rb'))
    data_out = {}

    # step 2. for each contributors, 
    #   get embeddings from model (in ApiTextExtractor). 
    klee = ApiTextExtractor()
    for contri_id, apis in tqdm(data_in.items()): 
        apis = [klee.api2vector(api) for api in apis]
        if len(apis) == 0: continue
        data_out[contri_id] = [sum(it) / len(apis) for it in zip(*apis)]


    # step 3. convert the result from 'idx 2 api(vector)' to 'contributor name 2 api(vector)'
    # step 4. assign random vector to unseen contributors.
    data_in = data_out
    data_out: Dict[str, np.ndarray] = {}

    contr2idx: Dict[str, int] = load_contributor_index()

    for contr_name, contr_idx in tqdm(contr2idx.items()): 
        vec = data_in.get(contr_idx)
        if vec is None: 
            vec = np.random.rand(200) * 2 - 1
        if len(vec) != 200: 
            print('panic!')
            print(data_in.get(contr_idx))
            exit(1)
        data_out[contr_name] = vec

    pickle.dump(
            data_out, 
            open(cfg['alpha']['embedding']['contributor_api_embedding'], 'wb'))


contributor_api_to_embedding()
