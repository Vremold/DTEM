import os
import json
import sys
import random

from config import Crawler
from path import CRAWLED_DIR, LOG_DIR, SELECTED_REPO_PATH

class RepositoryCrawler(Crawler):
    def __init__(
        self, 
        log_file=LOG_DIR+"respository_err.log"):
        super(RepositoryCrawler, self).__init__(log_file)

    def crawl_with_github_url(self, gh_url, auth_token=None):
        owner, repo = self.get_owner_and_repo_from_gh_url(gh_url)
        return self.crawl(owner, repo, auth_token)

    def crawl_with_target_url(self, url, auth_token=None):
        print("Now crawling URL: {}".format(url))

        response = self.request(url, auth_token)
        if not response:
            self.err_handling(url)
            return {}
        return response.json()
    
    def crawl(self, owner, repo, auth_token=None):
        url = "https://api.github.com/repos/{}/{}".format(owner, repo)
        return self.crawl_with_target_url(url, auth_token)

# ======================================================
"""
Repo Info Crawler
"""
def crawl_repo_info():
    repo_crawler = RepositoryCrawler()
    skipped_repos = set()
    with open(CRAWLED_DIR+"repo_statistics.txt", "r", encoding="utf-8") as inf:
        for line in inf:
            try:
                obj = json.loads(line)
            except:
                print(line)
                sys.exit(0)
            skipped_repos.add(obj.get("full_name", "").lower())
    
    repo_info_outf = open(CRAWLED_DIR+"repo_statistics.txt", "a+", encoding="utf-8")
    
    with open(SELECTED_REPO_PATH, "r", encoding="utf-8") as inf:
        repo_names = json.load(inf)
        for repo_name in repo_names:
            if repo_name.lower() in skipped_repos:
                continue

            owner, repo = repo_name.split("/")
            repo_info = repo_crawler.crawl(owner=owner, repo=repo)

            if repo_info:
                repo_info_outf.write(json.dumps(repo_info)+"\n")

if __name__ == "__main__":
    crawl_repo_info()