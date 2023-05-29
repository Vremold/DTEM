import os
import sys
import json

PR_COMMIT_FILE = "../../GHCrawler/cleaned/repo_pr_commits.txt"
DST_FILE = "./pr_modified_paths.json"

valid_suffix = [
    ".py",
    ".java",
    ".js",
    ".php",
    ".go",
    ".rb"
]

def contain_valid_suffix(fname:str):
    for s in valid_suffix:
        if fname.endswith(s):
            return True
    return False

if __name__ == "__main__":
    pr_modified_filenames = {}
    with open(PR_COMMIT_FILE, "r", encoding="utf-8") as inf:
        for line in inf:
            repo_name, pr_number, commit_url, contents = line.strip().split("\t")
            contents = json.loads(contents)
            pr_name = "{}##{}".format(repo_name, pr_number)
            if pr_name not in pr_modified_filenames:
                pr_modified_filenames[pr_name] = set()
            for c in contents:
                if contain_valid_suffix(c["filename"]):
                    pr_modified_filenames[pr_name].add(c["filename"])

    for pr_name in pr_modified_filenames:
        pr_modified_filenames[pr_name] = list(pr_modified_filenames[pr_name])

    new_pr_modified_filenames = {}
    for pr_name in pr_modified_filenames:
        filtered_fnames = [t for t in pr_modified_filenames[pr_name] if contain_valid_suffix(t)]
        if not filtered_fnames:
            continue
        new_pr_modified_filenames[pr_name] = filtered_fnames
    with open(DST_FILE, "w", encoding="utf-8") as outf:
        json.dump(new_pr_modified_filenames, outf, ensure_ascii=False)