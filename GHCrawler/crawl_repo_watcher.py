import os
import json
import sys
import time
import random
import shutil
import signal
import subprocess

from config import Crawler
from path import CRAWLED_DIR, LOG_DIR, SELECTED_REPO_PATH

class WatcherCrawler(Crawler):
    def __init__(
        self, 
        log_file=LOG_DIR+"watcher_err.log"):
        super(WatcherCrawler, self).__init__(log_file)

    def crawl(self, usr, repo, auth_token=None):
        results = list()
        page = 0
        per_page = 50
        while True:
            page += 1
            url = "https://api.github.com/repos/{}/{}/subscribers?per_page={}&page={}".format(usr, repo, per_page, page)
            response = self.request(url, auth_token)
            if not response:
                break
            stargazers = response.json()
            results.extend([x["login"] for x in stargazers])
        return results


# ======================================================
"""
Repo Watcher Crawler
"""
def crawl_watcher_info():
    crawler = WatcherCrawler()
    crawled_repos = set()
    if os.path.exists(CRAWLED_DIR+"repo_watchers.txt"):
        with open(CRAWLED_DIR+"repo_watchers.txt", "r", encoding="utf-8") as inf:
            for line in inf:
                repo_name, _ = line.strip().split("\t")
                crawled_repos.add(repo_name.lower())
    
    outf = open(CRAWLED_DIR+"repo_watchers.txt", "a+", encoding="utf-8")
    
    with open(SELECTED_REPO_PATH, "r", encoding="utf-8") as inf:
        repo_names = json.load(inf)
        for repo_name in repo_names:
            if repo_name.lower() in crawled_repos:
                continue
            
            print("Now processing repository: {}".format(repo_name))
            owner, repo = repo_name.split("/")
            watchers = crawler.crawl(usr=owner, repo=repo)

            outf.write("{}\t{}\n".format(repo_name, json.dumps(watchers, ensure_ascii=False)))

if __name__ == "__main__":
    crawl_watcher_info()