import os
import sys
import json

valid_suffix = [
    ".py",
    ".java",
    ".js",
    ".php",
    ".go",
    ".rb"
]

def is_valid(path:str):
    for s in valid_suffix:
        if path.endswith(s):
            return True
    return False

def contain_valid_paths(paths:list):
    for p in paths:
        if is_valid(p):
            return True
    return False

with open("./pr_modified_paths.json", "r", encoding="utf-8") as inf:
    pr_modified_paths = json.load(inf)
    valid_cnt = 0
    for pr in pr_modified_paths:
        # print(pr)
        m_paths = pr_modified_paths[pr]
        # print(m_paths)
        if contain_valid_paths(m_paths):
            valid_cnt += 1
        else:
            print(m_paths)

print("total prs:", len(pr_modified_paths))
print("total valid prs:", valid_cnt)