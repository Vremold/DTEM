import os
import json
import sys
import time
import random
import shutil
import signal
import subprocess
import base64

from config import Crawler
from path import LOG_DIR, CRAWLED_DIR, SELECTED_REPO_PATH, REDAME_DIR

class ReadmeCrawler(Crawler):
    def __init__(
        self, 
        log_file=LOG_DIR+"readme_err.log"):
        super(ReadmeCrawler, self).__init__(log_file)

    def crawl(self, usr, repo, auth_token=None):
        suffixs = [
            "README.md",
            "readme.md",
            "README.rst",
            "readme.rst",
            "README.txt",
            "readme.txt",
            "README",
            "readme",
            "README.markdown",
            "README.html",
            "README.htm",
            "readme.markdown",
            "readme.html",
            "readme.htm",
        ]
        # print(usr, repo)
        print(f"{usr}/{repo}")
        response = None
        for s in suffixs:
            url = "https://api.github.com/repos/{}/{}/contents/{}".format(usr, repo, s)
            print(url)
            response = self.request(url, auth_token)
            if response:
                break

        if not response:
            self.err_handling(f"{usr}/{repo}")
            return None
        response = response.json()
        if "content" not in response:
            return None
        content = base64.b64decode(response["content"])
        with open("./readme/{}#{}.md".format(usr, repo), "wb") as outf:
            outf.write(content)

    
    def crawl_error_repos(self, usr, repo, auth_token=None):
        content_url = "https://api.github.com/repos/{}/{}/contents".format(usr, repo)
        response = self.request(content_url, auth_token)
        if not response:
            return None
        response = response.json()
        print(f"{usr}/{repo}")
        fname = [(i["name"], i["path"]) for i in response]
        for name, path in fname:
            if "readme" in name or "README" in name:
                url = "https://api.github.com/repos/{}/{}/contents/{}".format(usr, repo, path)
                response = self.request(url, auth_token)
                if not response:
                    return None
                response = response.json()
                if "content" not in response:
                    return None
                content = base64.b64decode(response["content"])
                with open("./readme/{}#{}.md".format(usr, repo), "wb") as outf:
                    outf.write(content)
                return None
            
        
# ======================================================
"""
Repo README Crawler
"""
def crawl_readme():
    error_repos = set()
    with open(LOG_DIR+"readme_err.log", "r", encoding="utf-8") as inf:
        for line in inf:
            error_repos.add(line.strip())
    crawled_repos = set()
    for fname in os.listdir(REDAME_DIR):
        crawled_repos.add(fname[:-3].replace("#", "/"))
    crawler = ReadmeCrawler()
    with open(SELECTED_REPO_PATH, "r", encoding="utf-8") as inf:
        repos = json.load(inf)

    repos = repos[:20000]
    for repo in repos:
        if repo in crawled_repos or repo in error_repos:
            continue
        
        owner, repo_name = repo.split("/")
        crawler.crawl(owner, repo_name)

if __name__ == "__main__":
    crawl_readme()