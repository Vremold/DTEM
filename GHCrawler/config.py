import os
import sys
import json
import random
import time
import signal

import requests

auth_tokens = [
    "example_tokens"
]

suffixs = [
    ".py", ".ipynb",    # python 
    ".java",            # Java
    ".js",              # JavaScript
    ".php",             # PHP
    ".rb",              # Ruby
    ".go",              # Go
    ]

ignore_prefixs = [
    ".github",
    "build",
    ".vscode",
    ".docs",
    ".doc",
    ".licenses",
    ".license",
    "dist"
]

def set_timeout(num):
    def wrap(func):
        def handle(signum, frame):
            raise RuntimeError
 
        def to_do(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)  # 设置信号和回调函数
                signal.alarm(num)  # 设置 num 秒的闹钟
                r = func(*args, **kwargs)
                signal.alarm(0)  # 关闭闹钟
                return r
            # 超时返回
            except RuntimeError as e:
                return None
 
        return to_do
 
    return wrap

class Crawler(object):
    def __init__(self, log_file):
        self.errlog = open(log_file, "a+", encoding="utf-8")

    def __del__(self):
        self.errlog.close()

    @set_timeout(180)
    def request(self, url, auth_token=None, retry=2):
        if not auth_token:
            auth_token = random.choice(auth_tokens)
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": "token {}".format(auth_token)
        }
        while retry:
            try:
                response = requests.get(url, headers=headers)
                print(response.headers)
                print(response)
            except requests.exceptions.ProxyError as e:
                print("爬取速度过快了，休息一分钟")
                time.sleep(60)
            except requests.exceptions.SSLError as e:
                print("爬取速度过快了，休息一分钟")
                time.sleep(60)
            except requests.exceptions.ConnectionError as e:
                print("爬取速度过快了，休息一分钟")
                time.sleep(60)
            else:
                if response.status_code == 200:
                    return response
                else:
                    return None
            time.sleep(random.randint(1, 3))
            retry -= 1
        return None
    
    def get_owner_and_repo_from_gh_url(self, gh_url:str):
        last_idx = gh_url.rfind("/")
        repo = gh_url[last_idx+1:]
        last_second_idx = gh_url[:last_idx].rfind("/")
        owner = gh_url[last_second_idx+1:last_idx]
        return owner, repo

    def get_gh_url_from_owner_and_repo(self, owner, repo):
        return "https://github.com/{}/{}".format(owner, repo)
    
    def err_handling(self, url):
        self.errlog.write(url+"\n")
        pass