import os
import json
import sys
import time
import random
import shutil
import signal
import subprocess

from config import Crawler
from path import LOG_DIR, CRAWLED_DIR, SELECTED_REPO_PATH

class IssueCralwer(Crawler):
    def __init__(self, log_file=LOG_DIR+"pr_err.log"):
        super().__init__(log_file)
    
    def crawl(self, usr, repo, auth_token=None):
        results = []
        per_page = 50
        page = 0
        should_continue = True
        while should_continue:
            page += 1
            url = "https://api.github.com/repos/{}/{}/issues?per_page={}&page={}&state=all&since=2022-01-01T00:00:00Z".format(usr, repo, per_page, page)
            response = self.request(url, auth_token)
            if not response:
                should_continue = False
                return results
            issues = response.json()
            for p in issues:
                p_no = int(p["number"])
                # print(p_no)
                p_usr = p["user"]["login"]
                p_state = p["state"]
                p_body = p["body"]

                results.append({
                    "number": p_no,
                    "committer": p_usr,
                    "state": p_state,
                    "body": p_body,
                })
        return results

def crawl_repo_issue():
    pc = IssueCralwer()
    crawled_repos = set()
    if os.path.exists(CRAWLED_DIR+"repo_issues.txt"):
        with open(CRAWLED_DIR+"repo_issues.txt", "r", encoding="utf-8") as inf:
            for line in inf:
                repo, _ = line.strip().split("\t")
                crawled_repos.add(repo)
    
    outf = open(CRAWLED_DIR+"repo_issues.txt", "a+", encoding="utf-8")
    with open(SELECTED_REPO_PATH, "r", encoding="utf-8") as inf:
        repo_names = json.load(inf)
        for repo_name in repo_names:
            if repo_name in crawled_repos:
                continue
            
            print("Now processing repository: {}".format(repo_name))
            owner, repo = line.strip().split("/")
            prs = pc.crawl(owner, repo)
            outf.write("{}\t{}\n".format(repo_name, json.dumps(prs, ensure_ascii=False)))


if __name__ == "__main__":
    crawl_repo_issue()
    pass
