import os
import json
import sys
import time
import random
import shutil
import signal
import subprocess

from config import auth_tokens
from path import REPO_CODE_DIR, SELECTED_REPO_PATH, TMP_DIR, LOG_DIR

class ProjectCleaner(object):
    def __init__(self, 
                 suffixes=[".py", ".ipynb", ".java", ".js", ".php", ".rb", ".go", ".json", "pom.txt", ".md"]):
        self.suffixes = suffixes
        self.exclude_suffixes = ["node_modules", "venv", "_env", "_venv", "-venv", "-env", "vendor", "Recovery"]
    
    def if_match_suffix(self, filename):
        if not self.suffixes:
            return True
        
        for s in self.suffixes:
            if filename.endswith(s):
                return True
            
        return False
    
    def if_exclude(self, filename):
        for es in self.exclude_suffixes:
            if (filename.endswith(es)):
                return True
        return False
    
    def clean(self, src_project_dir, dst_project_dir, relative_paths=[], depth=0):
        if depth >= 10:
            return
        
        relative_path = "/".join(relative_paths)
        if self.if_exclude(relative_path):
            return
        
        curr_path = os.path.join(src_project_dir, relative_path)
        if os.path.isfile(curr_path):
            if not self.if_match_suffix(curr_path):
                return
            filename = relative_path.replace("/", "\\")
            try:
                shutil.copy(curr_path, os.path.join(dst_project_dir, filename))
            except OSError as e:
                # Too long file name
                if e.errno == 36:
                    shutil.copy(curr_path, os.path.join(dst_project_dir, "TOOLONG\\"+relative_paths[-1]))
                return
            return
        if os.path.islink(curr_path):
            return
        if os.path.isdir(curr_path):
            for filename in os.listdir(curr_path):
                self.clean(src_project_dir, dst_project_dir, relative_paths + [filename], depth+1)

pc = ProjectCleaner()

# ======================================================
"""
Repo Code Crawler
"""

def set_timeout(num):
    def wrap(func):
        def handle(signum, frame):
            raise RuntimeError
 
        def to_do(*args, **kwargs):
            try:
                signal.signal(signal.SIGABRT, handle)  # 设置信号和回调函数
                signal.alarm(num)  # 设置 num 秒的闹钟
                r = func(*args, **kwargs)
                signal.alarm(0)  # 关闭闹钟
                return r
            # 超时返回
            except RuntimeError as e:
                return False
 
        return to_do
 
    return wrap

@set_timeout(240)
def clone(owner, repo, dst_repo_content_dir):
    random_user_token = random.choice(auth_tokens)
    print("##Using Officail GitHub website[{}]".format(random_user_token))
    command = "git clone --depth=1 https://{}@github.com/{}/{} {}/{}#{}".format(random_user_token, owner, repo, dst_repo_content_dir, owner, repo)

    succeed = subprocess.call(command, shell=True)
    return succeed == 0

def clean_project(project_dir, owner, repo, dst_project_dir):
    src_project_dir = os.path.join(project_dir, "{}#{}".format(owner, repo))
    dst_project_dir = os.path.join(dst_project_dir, "{}#{}".format(owner, repo))
    if not os.path.exists(dst_project_dir):
        os.mkdir(dst_project_dir)
    pc.clean(src_project_dir, dst_project_dir, relative_paths=[], depth=0)
    shutil.rmtree(src_project_dir)

def clone_repository(tasks):
    log = open(os.path.join(LOG_DIR, "repo_code_err.log"), "a+", encoding="utf-8")
    
    if not os.path.exists(REPO_CODE_DIR):
        os.mkdir(REPO_CODE_DIR)
    
    for repo_name in tasks:
        owner, repo = repo_name.split("/")
        succeed = clone(owner, repo, TMP_DIR)
        if not succeed:
            log.write("{}/{}\n".format(owner, repo))
        else:
            clean_project(TMP_DIR, owner, repo, REPO_CODE_DIR)

def get_tasks(task_file):
    crawled_repos = set()
    for filename in os.listdir(REPO_CODE_DIR):
        if "#" not in filename:
            continue
        owner, repo = filename.split("#")
        crawled_repos.add("{}/{}".format(owner, repo))
    
    tasks = set()
    with open(task_file, "r", encoding="utf-8") as inf:
        tasks = set(json.load(inf))
    tasks = tasks - crawled_repos

    return tasks

if __name__ == "__main__":
    clone_repository(get_tasks(SELECTED_REPO_PATH))
    pass
