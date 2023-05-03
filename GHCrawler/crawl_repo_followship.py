import os
import json
import sys
import time
import random
import shutil
import signal
import subprocess

from config import auth_tokens, Crawler
from path import CRAWLED_DIR, LOG_DIR

class FollowerShipCrawler(Crawler):
    def __init__(
        self, 
        log_file=LOG_DIR+"follower.log"):
        super(FollowerShipCrawler, self).__init__(log_file)
    
    def crawl(self, usr_name, auth_token=None):
        results = []
        per_page = 50
        page = 1
        while True:
            # davidB
            url = "https://api.github.com/users/{}/followers?per_page={}&page={}".format(usr_name, per_page, page)
            response = self.request(url, auth_token)
            if not response:
                return results

            followers = response.json()
            if followers:
                results.extend([item["login"] for item in followers])
                page += 1
            else:
                break
        return results


class FollowingShipCrawler(Crawler):
    def __init__(
        self, 
        log_file=LOG_DIR+"following.log"):
        super(FollowingShipCrawler, self).__init__(log_file)
    
    def crawl(self, usr_name, auth_token=None):
        results = []
        per_page = 50
        page = 1
        while True:
            url = "https://api.github.com/users/{}/following?per_page={}&page={}".format(usr_name, per_page, page)
            response = self.request(url, auth_token)
            if not response:
                return results
            
            followings = response.json()
            if followings:
                results.extend([item["login"] for item in followings])
                page += 1
            else:
                break
        return results

# ======================================================
"""
FollowerShip Crawler
"""
def crawl_followership():
    followership_crawler = FollowerShipCrawler()
    crawled_usrs = set()
    if os.path.exists(CRAWLED_DIR+"user_followers.txt"):
        with open(CRAWLED_DIR+"user_followers.txt", "r", encoding="utf-8") as inf:
            for line in inf:
                usr, followers = line.strip().split("\t")
                crawled_usrs.add(usr)
    
    outf = open(CRAWLED_DIR+"user_followers.txt", "a+", encoding="utf-8")
    # "contributors.json" should be generated from the results of  "crawl_contributors.py"
    with open("contributor_nodes.json", "r", encoding="utf-8") as inf:
        usrs = json.load(inf)
        for usr in usrs:
            if usr in crawled_usrs:
                continue
            print("Now processing user: ", usr)
            followers = followership_crawler.crawl(usr)
            if followers:
                outf.write("{}\t{}\n".format(usr, json.dumps(followers)))

# ======================================================
"""
FollowingShip Crawler
"""
def crawl_followingship():
    followingship_crawler = FollowingShipCrawler()
    crawled_usrs = set()
    if os.path.exists(CRAWLED_DIR+"user_followings.txt"):
        with open(CRAWLED_DIR+"user_followings.txt", "r", encoding="utf-8") as inf:
            for line in inf:
                usr, followings = line.strip().split("\t")
                crawled_usrs.add(usr)
    
    outf = open(CRAWLED_DIR+"user_followings.txt", "a+", encoding="utf-8")
    # "contributors.json" should be generated from the results of  "crawl_contributors.py"
    with open("contributors.json", "r", encoding="utf-8") as inf:
        usrs = json.load(inf)
        for usr in usrs:
            if usr in crawled_usrs:
                continue
            print("Now processing user: ", usr)
            followings = followingship_crawler.crawl(usr)
            if followings:
                outf.write("{}\t{}\n".format(usr, json.dumps(followings)))

if __name__ == "__main__":
    crawl_followingship()
