import os
import json
import sys
import time
import random
import shutil
import signal
import subprocess

from config import Crawler, auth_tokens
from path import LOG_DIR, CRAWLED_DIR, SELECTED_REPO_PATH

class RepoLanguageCrawler(Crawler):
    def __init__(
        self, 
        log_file=LOG_DIR+"language_err.log"):
        super().__init__(log_file)

    def crawl(self, usr, repo, auth_token=None):
        url = "https://api.github.com/repos/{}/{}/languages".format(usr, repo)
        response = self.request(url, auth_token)
        if not response:
            self.err_handling(url)
            return None
        return response.json()


# ======================================================
"""
Repo Language Crawler
"""
def crawl_repo_languages():
    cralwer = RepoLanguageCrawler()
    outf = open(CRAWLED_DIR+"repo_languages.txt", "a+", encoding="utf-8")
    with open(SELECTED_REPO_PATH, "r", encoding="utf-8") as inf:
        repo_names = json.load(inf)
    for repo_name in repo_names:
        print("Now processing repository: {}".format(repo_name))
        owner, repo = repo_name.strip().split("/")
        languages = cralwer.crawl(usr=owner, repo=repo, auth_token=random.choice(auth_tokens))
        if not languages:
            continue
        outf.write("{}\t{}\n".format(repo_name, json.dumps(languages, ensure_ascii=False)))
    outf.close()

if __name__ == "__main__":
    crawl_repo_languages()