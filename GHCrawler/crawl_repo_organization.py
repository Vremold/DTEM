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

class OrganizationCrawler(Crawler):
    def __init__(
        self, 
        log_file=LOG_DIR+"organization.log"):
        super(OrganizationCrawler, self).__init__(log_file)
    
    def crawl(self, usr_name, auth_token=None):
        per_page = 50
        url = "https://api.github.com/users/{}/orgs?per_page={}".format(usr_name, per_page)
        response = self.request(url, auth_token)
        if not response:
            return []

        orgs = response.json()
        return [x["login"] for x in orgs]

# ======================================================
"""
Organization Crawler
"""
def crawl_organization():
    crawler = OrganizationCrawler()
    skipped_usrs = set()
    if os.path.exists(CRAWLED_DIR+"user_organizations.txt"):
        with open(CRAWLED_DIR+"user_organizations.txt", "r", encoding="utf-8") as inf:
            for line in inf:
                usr, _ = line.strip().split("\t")
                skipped_usrs.add(usr)
    
    outf = open(CRAWLED_DIR+"user_organizations.txt", "a+", encoding="utf-8")
    # "contributors.json" should be generated from the results of  "crawl_contributors.py"
    with open("contributor_nodes.json", "r", encoding="utf-8") as inf:
        usrs = json.load(inf)
        for usr in usrs:
            if usr in skipped_usrs:
                continue
            print("Now processing user: ", usr)
            orgs = crawler.crawl(usr)
            if orgs:
                outf.write("{}\t{}\n".format(usr, json.dumps(orgs, ensure_ascii=False)))

if __name__ == "__main__":
    crawl_organization()