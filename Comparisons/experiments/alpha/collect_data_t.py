#!/usr/bin/env python

from rich import print as rprint
from Comparisons.experiments.general import \
    ignore_exception, \
    load_yaml_cfg, \
    load_contributor_index, load_repository_index, load_issue_index, \
    dict_invert, dict_inspect, dict_invert_mul, \
    github_token, \
    load_jsonl, save_jsonl, \
    data_divide, \
    RepoDict
from typing import Dict, List, Any, Optional, TypedDict, Generator, Tuple
from github import Github
from github.Repository import Repository
from tqdm import tqdm
import os, sys

'''
    收集数据. 
    原本的项目是以仓库为 key 组织的. 
    但我们想要复现的这个论文, 是以开发者为 key 组织的各种数据. 

    我们将 repo, issue, api 的信息, 组织起来, 
    输出到 Comparisons/data/alpha 下. 
'''

# Repository:   Name, Tags, Topics, README
# Issue:        Body & Title
# API:          (appear at least 5 times)

CONFIG_FILE_PATH = 'Comparisons/experiments/config.yaml'
cfg: Dict[str, Any] = load_yaml_cfg()['alpha']



class RepoDataCrawler(): 
    '''
        爬出来的数据并不是齐全的. 
        有 readme 的文件的数量是 44574 / 50k; 
        有 topics 和 tags 的文件的数量是 49709 / 50k. 
        所以后续注意, 要为这些没有的信息赋予初值. 
    '''

    repos: Dict[str, RepoDict]
    repo_names: Dict[int, str]

    def __init__(self):
        self.github = Github(github_token(2))
        self.repo_names = dict_invert(load_repository_index())
        self.repos = {}

    def fetch_all(self): 
        stdout = cfg['raw']['repo_file_path']
        self.repos = { it['name']: it for it in load_jsonl(stdout) }

        scope: List[Any] = list(self.repo_names.items())

        try: 
            for _, repo_name in tqdm(scope): 
                if repo_name in self.repos: 
                    continue
                @ignore_exception
                def _():
                    self.get_repo_info(repo_name)
        finally: 
            save_jsonl(stdout, self.repos.values())


    def get_repo_info(self, repo_name: str) -> RepoDict: 
        '''
            获取的结果会同时返回给调用方, 
            也会添加到 repos 字典中. 
        '''
        if repo_name in self.repos: 
            return self.repos[repo_name]
        
        repo: Repository = self.github.get_repo(repo_name)

        repo = {
            'name':     repo_name,
            'tags':     self._get_repo_tag(repo),
            'topic':    self._get_repo_topic(repo),
        }

        self.repos[repo_name] = repo
        return repo


    def _get_repo_tag(self, repo: Repository) -> List[str]: 
        return repo.get_topics()


    def _get_repo_topic(self, repo: Repository) -> str: 
        return repo.description

class IssueDataCrawler: 

    '''
        可惜的是, 当时学长在爬取数据的时候没有把标题也爬取下来,
        所以需要我们重新爬取. 

        在数据源中, 我们已经有了 issue 所属仓库和 id. 
        为了降低访问 github 的次数, 我们按照 repo 先将所有的 issue 聚合, 
        然后再依次爬取. 
    '''

    repo_issue_indices: Dict[str, List[int]]

    def __init__(self): 
        ret = {}
        for issue_name in load_issue_index():
            tmp = issue_name.split('#')
            repo_name, issue_idx = tmp[0], int(tmp[1])

            if repo_name not in ret: 
                ret[repo_name] = []
            ret[repo_name].append(issue_idx)

        self.repo_issue_indices = ret

    def fetch_all(self, idx: int, total=4, gh_token_idx=None): 
        '''
            idx 和 total 是用来选择数据的: 选择total份中的第idx份数据来处理. 
        '''

        # jsonl file
        # elem looks like: {'name': 'datalux/osintgram#670', 'title': '...'}
        stdout = cfg['raw']['issue_title_file_path'] + f'.{idx}'
        klee = Github(github_token(gh_token_idx))
        crawled_issues: List[str] = \
            list(it['name'] for it in load_jsonl(stdout))
        
        data = list(data_divide(self.repo_issue_indices.items(), idx, total))
        for repo_name, issue_indices in tqdm(data): 
            ret = []
            try: 
                @ignore_exception
                def _():
                    repo: Repository = klee.get_repo(repo_name)
                    for idx in tqdm(issue_indices): 
                        issue_name = f'{repo_name}#{idx}'
                        if issue_name in crawled_issues: 
                            continue
                        # print(f'self.get_issue_title({repo}, {idx})')
                        title = self.get_issue_title(repo, idx)
                        ret.append({
                            'name':     issue_name,
                            'title':    title,
                        })
            finally:
                save_jsonl(stdout, ret, 'a')

    def get_issue_title(self, repo: Repository, issue_id: int) -> str:
        return repo.get_issue(issue_id).title

idx = int(sys.argv[1])

# klee = RepoDataCrawler()
# klee.fetch_all()

klee = IssueDataCrawler()
klee.fetch_all(idx, total=4, gh_token_idx=idx)
