# Data Structure
## user_organizations.txt
user`\t`organzations
## user_followings.txt
user`\t`followings
## user_followers.txt
user`\t`followers
## repo_watchers.txt
repo`\t`watchers
## repo_stargazers.txt 
repo`\t`stargazers
## repo_languages.txt
```
repo`\t`{
    language: percentage (integer not float)
}
```
## repo_contributions.txt
```
repo`\t`[
    (contributor, contribution_count)
]
```
## repo_statistics.txt
```
{
    "name": "scala-maven-plugin", 
    "full_name": "davidB/scala-maven-plugin",
    "owner": {
        "login": "davidB"
    },
    "language": "Java",
    "stargazers_count": int,
    "topics": [""],
}
```
## repo_prs.txt
```
repo`\t`[
    {
        "number": p_no,
        "committer": p_user,
        "if_closed": p_closed,
        "if_merged": p_merged,
        "body": p_body,
        "commit_urls": p_commit_urls,
        "reviewers": p_requested_reviewers
    }
]
```
## repo_issues.txt
```
repo`\t`[
    {
        "number": p_no,
        "committer": p_usr,
        "state": p_state,
        "body": p_body,
    }
]
```
## repo_pr_commits.txt
```
repo`\t`pr_no`\t`commit_url`\t`[
    {
        "filename": filename,
        "patch": patch,
        "content_url": content_url,
        "raw_url": raw_url
    }
]
```

# Attention
实际上本文获取的数据并不止项目提到的50000，大约应有80000个仓库，但是由于获取仓库的Pull Request和Issue过于花费时间，本文最终选择数据是50000个。