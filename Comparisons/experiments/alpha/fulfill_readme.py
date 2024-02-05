#!/usr/bin/env python3

'''
    先不要运行这个脚本.
    这个脚本一开始是放在根目录下的, 目的是补全 README 文件. 
    这个任务在对比实验 alpha 中, 用来生成仓库的嵌入表达. 

    这个脚本似乎会将生成的 readme 文件放在根目录下的一个叫 readme 的目录中, 
    然后这些文件被手动移动到和已有 readme 文件一起的位置上. 

    我 disable 了这个脚本, 但这个脚本仍然可读, 希望对你理解对比试验的其他部分有所帮助. 
'''

'''
    原有的代码数据集的 readme 文件不全, 有那么 5426 个仓库没有README. 
    我们将这些额外的仓库的内容抽取下来, 补充到原来的目录下. 
'''


exit(0)

from Comparisons.experiments.general import \
    load_yaml_cfg, github_token, \
    data_divide
import os
from tqdm import tqdm
import sys
from github import Github


cfg = load_yaml_cfg()['alpha']['collect_data']

possible_readme_names = [
    "Readme.md", 
    "README.md",        "readme.md",
    "README.textile",   "readme.textile", 
    "README.adoc",      "readme.adoc",
    "README.rst",       "readme.rst",
    "README.txt",       "readme.txt",
    "README",           "readme",
    "README.markdown",  "readme.markdown",
    "README.html",      "readme.html", 
    "README.htm",       "readme.htm",
]  # copied from `GHCrawler/crawl_repo_readme.py`


def crawl(idx: int, gh_token_idx=-1, total=4):

    repos = list(data_divide([
        it.strip().replace('#', '/') 
        for it in open(cfg['repo_without_readme_list_file'])],
    idx, total))

    klee = Github(github_token(gh_token_idx))

    all_readme_files = os.listdir(cfg['readme_directory'])

    for repo_name in tqdm(repos): 
        filename = f'{repo_name.replace("/", "#")}.md'
        if filename in all_readme_files: continue
        
        try: 
            repo = klee.get_repo(repo_name)
        except: continue


        for readme_name in possible_readme_names: 
            try: 
                print(f'trying {repo_name} => {readme_name}')
                content = repo.get_contents(readme_name)\
                        .decoded_content.decode('utf-8')

                with open(os.path.join(cfg['readme_directory'], filename), 'w') as fp: 
                    fp.write(content)
                break
            except KeyboardInterrupt: exit(0)
            except: continue


crawl(*map(int, sys.argv[1:]))
