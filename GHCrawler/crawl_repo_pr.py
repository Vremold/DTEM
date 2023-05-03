import os
import json
import sys
import time
import random
import shutil
import signal
import subprocess

from config import  Crawler
from path import LOG_DIR, CRAWLED_DIR, SELECTED_REPO_PATH

class PRCralwer(Crawler):
    def __init__(self, log_file=LOG_DIR+"/pr_err.log"):
        super().__init__(log_file)
    
    def crawl(self, usr, repo, auth_token=None):
        results = []
        per_page = 50
        page = 0
        should_continue = True
        while should_continue:
            page += 1
            url = "https://api.github.com/repos/{}/{}/pulls?per_page={}&page={}&state=all".format(usr, repo, per_page, page)
            response = self.request(url, auth_token)
            if not response:
                should_continue = False
                return results
            pulls = response.json()
            for p in pulls:
                # created at 2022 or 2023
                if not p["created_at"].startswith("2022") and not p["created_at"].startswith("2023"):
                    should_continue = False
                    break
                
                p_no = int(p["number"])
                p_user = p["user"]["login"]
                # print(p_no)
                p_closed = p["closed_at"]
                p_merged = p["merged_at"]
                p_body = p["body"]
                p_commit_urls = list()
                response = self.request(p["commits_url"], auth_token)
                if response:
                    p_commit_urls = [x["url"] for x in response.json()]
                p_requested_reviewers = p["requested_reviewers"]

                results.append({
                    "number": p_no,
                    "committer": p_user,
                    "if_closed": p_closed,
                    "if_merged": p_merged,
                    "body": p_body,
                    "commit_urls": p_commit_urls,
                    "reviewers": p_requested_reviewers
                })
        return results

def crawl_repo_pr():
    pc = PRCralwer()
    crawled_repos = set()
    if os.path.exists(CRAWLED_DIR+"repo_prs.txt"):
        with open(CRAWLED_DIR+"repo_prs.txt", "r", encoding="utf-8") as inf:
            for line in inf:
                repo, _ = line.strip().split("\t")
                crawled_repos.add(repo)
    
    outf = open(CRAWLED_DIR+"repo_prs.txt", "a+", encoding="utf-8")
    with open(SELECTED_REPO_PATH, "r", encoding="utf-8") as inf:
        repo_names = json.load(inf)
        for repo_name in repo_names:
            if repo_name in crawled_repos:
                continue
            
            print("Now processing repository: {}".format(repo_name))
            owner, repo = repo_name.split("/")
            prs = pc.crawl(owner, repo)
            outf.write("{}\t{}\n".format(repo_name, json.dumps(prs, ensure_ascii=False)))


if __name__ == "__main__":
    crawl_repo_pr()
    pass