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

class ContributorCrawler(Crawler):
    def __init__(
        self, 
        per_page,
        log_file=LOG_DIR+"/contributor_err.log"):

        self.per_page = per_page
        super(ContributorCrawler, self).__init__(log_file)

    def crawl_with_github_url(self, gh_url, auth_token=None):
        owner, repo = self.get_owner_and_repo_from_gh_url(gh_url)
        return self.crawl(owner, repo, auth_token)
    
    def crawl_with_target_url(self, url, auth_token=None, contributors_bound_limit=10):
        print("Now crawling URL: {}".format(url))
        results, should_continue = list(), True
        response = self.request(url, auth_token)
        if not response or not response.json():
            self.err_handling(url)
            return results, False

        for item in response.json():
            if item["contributions"] < contributors_bound_limit:
                should_continue = False
                break
            else:
                results.append([item["login"], item["contributions"]])
        return results, should_continue
    
    def crawl(self, owner, repo, auth_token=None, contributors_bound_limit=10):
        page_no = 0
        should_continue = True
        results = list()
        while should_continue:
            page_no += 1
            url = "https://api.github.com/repos/{}/{}/contributors?per_page={}&page={}".format(owner, repo, self.per_page, page_no)

            page_result, should_continue = self.crawl_with_target_url(url, auth_token, contributors_bound_limit=contributors_bound_limit)
            results.extend(page_result)
        return results

# ======================================================
"""
Repo Contributor Crawler
"""
def crawl_contributor_info():
    contri_crawler = ContributorCrawler(per_page=50)
    crawled_repos = set()
    with open(CRAWLED_DIR+"repo_contributions.txt", "r", encoding="utf-8") as inf:
        for line in inf:
            repo_name, _ = line.strip().split("\t")
            crawled_repos.add(repo_name.lower())
    
    repo_contri_outf = open(CRAWLED_DIR+"repo_contributions.txt", "a+", encoding="utf-8")
    
    with open(SELECTED_REPO_PATH, "r", encoding="utf-8") as inf:
        repo_names = json.load(inf)
        for repo_name in repo_names:
            if repo_name.lower() in crawled_repos:
                continue

            owner, repo = repo_name.split("/")
            contributors = contri_crawler.crawl(owner=owner, repo=repo)

            repo_contri_outf.write("{}\t{}\n".format(repo_name, json.dumps(contributors, ensure_ascii=False)))

if __name__ == "__main__":
    crawl_contributor_info()