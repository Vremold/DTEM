import os
import json
import time
import random
import csv
import re
import subprocess
import argparse

from crawler import RepositoryCrawler, ContentCrawler, ContributorCrawler

parser = argparse.ArgumentParser()
parser.add_argument("--lb", default=0, type=int, help="lower bound of crawling file range")
parser.add_argument("--ub", default=10, type=int, help="upper range ofcrawling file range")
parser.add_argument("--authtoken", type=str, help="GitHub personal access token")
parser.add_argument("--use_random_token", type=bool, default=True, help="If use random auth token")
args = parser.parse_args()

auth_tokens = [
    "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", # Vremold
    "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", # JaiqiZhang
    "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", # Xremold
]

# Choose the following six kinds of programming languages
# refered to CodeBert
suffixs = [
    ".py", ".ipynb",    # python 
    ".java",            # Java
    ".js",              # JavaScript
    ".php",             # PHP
    ".rb",              # Ruby
    ".go",              # Go
    ]

ignore_prefixs = [
    ".github",
    "build",
    ".vscode",
    ".docs",
    ".doc",
    ".licenses",
    ".license",
    "dist"
]

def crawl_repository_information(ghs_csv_file, dst_repo_stat_file, skip):
    rc = RepositoryCrawler()
    crawled_repos = set()
    if os.path.exists(dst_repo_stat_file):
        with open(dst_repo_stat_file, "r", encoding="utf-8") as inf:
            for line in inf:
                # print(line)
                obj = json.loads(line)
                crawled_repos.add(obj.get("full_name", "").lower())
    with open(ghs_csv_file, "r", encoding="utf-8") as inf, open(dst_repo_stat_file, "a+", encoding="utf-8") as outf:
        next(inf)
        csv_reader = csv.reader(inf)
        for line in csv_reader:
            if skip >= 0:
                skip -= 1
                continue
            repo_info = line[0]
            if repo_info.lower() in crawled_repos:
                continue

            # print("Now crawling repository {}".format(repo_info))
            owner, repo = repo_info.split("/")
            response = rc.crawl(owner, repo)
            outf.write(json.dumps(response, ensure_ascii=False)+"\n")

def crawl_repository_information_leaked(dst_repo_stat_file, log_file):
    rc = RepositoryCrawler()
    leaked_repo_urls = set()
    with open(log_file, "r", encoding="utf-8") as inf:
        leaked_repo_urls = set([line.strip() for line in inf.readlines()])
    with open(log_file, "w", encoding="utf-8") as outf:
        outf.truncate()
    with open(dst_repo_stat_file, "a+", encoding="utf-8") as outf:
        for url in leaked_repo_urls:
            response = rc.crawl_with_target_url(url)
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

def crawl_repository_contributors_leaked(dst_repo_contribution_file, log_file):
    def get_repo_owner_and_name_from_url(url):
        match = re.search(r"https://api.github.com/repos/(.*?)/(.*?)/", url)
        return match.group(1), match.group(2)
    contribution_crawler = ContributorCrawler(per_page=50)
    leaked_repo_urls = set()
    with open(log_file, "r", encoding="utf-8") as inf:
        leaked_repo_urls = set([line.strip() for line in inf.readlines()])
    with open(log_file, "w", encoding="utf-8") as outf:
        outf.truncate()
    with open(dst_repo_contribution_file, "a+", encoding="utf-8") as outf:
        for url in leaked_repo_urls:
            response = rc.crawl_with_target_url(url)
            outf.write(json.dumps(response, ensure_ascii=False)+"\n")
            owner, repo = get_repo_owner_and_name_from_url(url)
            outf.write("{}/{}\t{}\n".format(
                owner, repo, 
                json.dumps(response, ensure_ascii=False)
            ))


def crawl_repository_content(ghs_csv_file, dst_repo_content_file):
    crawler = ContentCrawler(suffixs=suffixs, ignored_prefixs=ignore_prefixs, per_page=50, cache_url="./log/crawled_content_urls.txt")

    with open(ghs_csv_file, "r", encoding="utf-8") as inf, open(dst_repo_content_file, "a+", encoding="utf-8") as outf:
        next(inf)
        csv_reader = csv.reader(inf)
        for line in csv_reader:
            repo_info = line[0]
            
            print("Now crawling repository {}".format(repo_info))
            owner, repo = repo_info.split("/")
            response = crawler.crawl(owner, repo, outf)

def crawl_repository_content_clone(idx_list, dst_repo_content_dir, log_file):

    def build_clone_command(owner, repo, dst_repo_content_dir):
        rand = random.randint(4, 6)
        random_user_token = random.choice(auth_tokens)
        if rand > 3:
            print("##Using Officail GitHub website[{}]".format(random_user_token))
            command = "git clone --depth=1 https://{}@github.com/{}/{} {}/{}#{}".format(random_user_token, owner, repo, dst_repo_content_dir, owner, repo)
        elif rand == 0:
            print("##Using GitHub Proxy: https://gitclone.com")
            command = "git clone --depth=1 https://gitclone.com/github.com/{}/{} {}/{}#{}".format(owner, repo, dst_repo_content_dir, owner, repo)
        # elif rand == 1:
        #     # https://ghproxy.com/
        #     command = "git clone --depth=1 https://ghproxy.com/https://github.com/{}/{} {}/{}#{}".format(owner, repo, dst_repo_content_dir, owner, repo)
        elif rand == 1:
            # https://hub.0z.gs/
            command = "git clone --depth=1 https://hub.0z.gs/{}/{} {}/{}#{}".format(owner, repo, dst_repo_content_dir, owner, repo)
            print("##Using GitHub proxy: https://hub.0z.gs")
        elif rand == 2:
            # https://hub.fastgit.xyz/
            command = "git clone --depth=1 https://hub.fastgit.xyz/{}/{} {}/{}#{}".format(owner, repo, dst_repo_content_dir, owner, repo)
            print("##Using GitHub proxy: https://hub.fastgit.xyz")
        elif rand == 3:
            # https://gh.api.99988866.xyz/
            command = "git clone --depth=1 https://gh.api.99988866.xyz/https://github.com/{}/{} {}/{}#{}".format(owner, repo, dst_repo_content_dir, owner, repo)
            print("##Using GitHub proxy: https://hub.fastgit.xyz")

        return command
    crawled_repos = set()
    for filename in os.listdir(dst_repo_content_dir):
        owner, repo = filename.split("#")
        crawled_repos.add("{}/{}".format(owner, repo))
    error_repos = set()
    with open(log_file, "r", encoding="utf-8") as inf:
        for line in inf:
            error_repos.add(line.strip())

    log = open(log_file, "a+", encoding="utf-8")
    if not os.path.exists(dst_repo_content_dir):
        os.mkdir(dst_repo_content_dir)
    
    for idx in idx_list:
        with open("./ghs/ghs_{}.csv".format(idx), "r", encoding="utf-8") as inf:
            next(inf)
            csv_reader = csv.reader(inf)
            for line in csv_reader:
                repo_info = line[0]
                if repo_info in crawled_repos or repo_info in error_repos:
                    continue
            
                owner, repo = repo_info.split("/")

                command = build_clone_command(owner, repo, dst_repo_content_dir)
                
                # os.system(command)
                succeed = subprocess.call(command, shell=True)
                if succeed != 0:
                    log.write("{}/{}\n".format(owner, repo))
                print("Crawling {}/{} finished".format(owner, repo))
                time.sleep(random.randint(5, 30))


if __name__ == "__main__":
    """ Crawl Repository Statistics"""
    # crawl_repository_information("./ghs.csv", "repo_statistics.txt")
    # crawl_repository_information_leaked("repo_statistics.txt", "./log/respository_err.log")

    """Crawl Repository Contributors Statistics"""
    # crawl_repository_contributors("./ghs.csv", "repo_contribution.txt")
    # crawl_repository_contributors_leaked("repo_contribution.txt", "contributor_err.log")

    """Crawl Repository Content Files"""
    crawl_repository_content_clone(range(args.lb, args.ub), "repo_content", "./log/content_err.log")
