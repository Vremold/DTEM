import os
import json
import sys
import time
import random
import shutil
import signal
import subprocess

from config import auth_tokens, Crawler
from path import LOG_DIR, CRAWLED_DIR

class PRCommitCralwer(Crawler):
    def __init__(self, log_file=os.path.join(LOG_DIR, "pr_commit_err.log")):
        super().__init__(log_file)
    
    def crawl(self, pr_commit_url, auth_token=None):
        results = []
        response = self.request(pr_commit_url)
        if not response:
            return []

        changed_files = response.json()["files"]
        n_changed_files = len(changed_files)

        filenames = [x["filename"] for x in changed_files]
        patches = [x.get("patch", "") for x in changed_files]
        content_urls = [x["contents_url"] for x in changed_files]
        raw_urls = [x["raw_url"] for x in changed_files]
        
        for filename, patch, content_url, raw_url in zip(filenames, patches, content_urls, raw_urls):
            results.append({
                "filename": filename,
                "patch": patch,
                "content_url": content_url,
                "raw_url": raw_url
            })
        return results

def crawl_repo_pr_commit():
    pc = PRCommitCralwer()

    outf = open(CRAWLED_DIR+"repo_pr_commits.txt", "a+", encoding="utf-8")
    with open(CRAWLED_DIR+"repo_prs.txt", "r", encoding="utf-8") as inf:
        for line in inf:
            repo_name, prs = line.strip().split("\t")
            prs = json.loads(prs)

            for pr in prs:
                p_no = pr["number"]
                
                for url in pr["commit_urls"]:
                    print("Now processing repository: {}, PR: {}, commit_url: {}".format(repo_name, p_no, url))
                    commit_codes = pc.crawl(url)
                    outf.write("{}\t{}\t{}\t{}\n".format(repo_name, p_no, url, json.dumps(commit_codes, ensure_ascii=False)))
    outf.close()

if __name__ == "__main__":
    crawl_repo_pr_commit()
