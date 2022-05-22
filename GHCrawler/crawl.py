import os
import json
import time
import random
import csv

from crawler import RepositoryCrawler, ContentCrawler, ContributorCrawler

suffixs = [
    ".js", "ts",        # JavaScript/TypeScript
    ".c", ".cpp",       # C/C++
    ".cs",              # c#
    ".py", ".ipynb",    # python 
    ".java",            # Java
    ".go",              # Go
    ".html", ".css"     # HTML/CSS
    ".md",              # others
    ]

def test():
    owner = "facebook"
    repo = "react"
    rc = RepositoryCrawler()
    content_crawler = ContentCrawler(suffixs, per_page=50)
    contributor_crawler = ContributorCrawler(per_page=50)
    # print(contributor_crawler.crawl(owner, repo))
    print(content_crawler.crawl(owner, repo, path="scripts"))

def crawl_repository_information(ghs_csv_file, dst_repo_stat_file):
    rc = RepositoryCrawler()
    crawled_repos = set()
    if os.path.exists(dst_repo_stat_file):
        with open(dst_repo_stat_file, "r", encoding="utf-8") as inf:
            for line in inf:
                obj = json.loads(line)
                crawled_repos.add(obj["full_name"])
    with open(ghs_csv_file, "r", encoding="utf-8") as inf, open(dst_repo_stat_file, "a+", encoding="utf-8") as outf:
        next(inf)
        csv_reader = csv.reader(inf)
        for line in csv_reader:
            repo_info = line[0]
            if repo_info in crawled_repos:
                continue
            
            # print("Now crawling repository {}".format(repo_info))
            owner, repo = repo_info.split("/")
            response = rc.crawl(owner, repo)
            outf.write(json.dumps(response, ensure_ascii=False)+"\n")

def crawl_repository_contributors(ghs_csv_file, dst_repo_contribution_file):
    contribution_crawler = ContributorCrawler(per_page=50)
    
    crawled_repos = set()
    if os.path.exists(dst_repo_contribution_file):
        with open(dst_repo_contribution_file, "r", encoding="utf-8") as inf:
            for line in inf:
                crawled_repos.add(line.split("\t")[0])
    
    with open(ghs_csv_file, "r", encoding="utf-8") as inf, open(dst_repo_contribution_file, "a+", encoding="utf-8") as outf:
        next(inf)
        csv_reader = csv.reader(inf)
        for line in csv_reader:
            repo_info = line[0]
            if repo_info in crawled_repos:
                continue
            
            # print("Now crawling repository {}".format(repo_info))
            owner, repo = repo_info.split("/")
            response = contribution_crawler.crawl(owner, repo)
            outf.write("{}\t{}\n".format(
                repo_info, 
                json.dumps(response, ensure_ascii=False)
            ))

if __name__ == "__main__":
    # crawl_repository_information("./ghs.csv", "repo_contribution.txt")
    crawl_repository_contributors("./ghs.csv", "repo_contribution.txt")