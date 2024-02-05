import json
import os
import sys
import random

"""
    这个类只在这个文件中调用一次. 
    其功能是在处理 GHCrawler 中爬取的数据: 

    ../../GHCrawler/cleaned/* => ./full_graph/content/*
"""

class CrawledDataLoader(object):
    def __init__(self, crawled_data_dir, processed_data_dir, follow_sample_percent=1/3) -> None:
        self.processed_data_dir = processed_data_dir
        self.crawled_data_dir = crawled_data_dir
        self.orgs = dict()
        self.contributors = dict()
        self.repos = dict()
        self.issues = dict()
        self.prs = dict()
        self.follow_sample_percent = follow_sample_percent
    
    def _get_repo_id(self, repo):
        if repo not in self.repos:
            self.repos[repo] = len(self.repos)
        return self.repos[repo]
    
    def _get_contributor_id(self, contributor):
        if contributor not in self.contributors:
            self.contributors[contributor] = len(self.contributors)
        return self.contributors[contributor]

    def _get_org_id(self, org):
        if org not in self.orgs:
            self.orgs[org] = len(self.orgs)
        return self.orgs[org]

    def _get_issue_id(self, issue):
        if issue not in self.issues:
            self.issues[issue] = len(self.issues)
        return self.issues[issue]
    
    def _get_pr_id(self, pr):
        if pr not in self.prs:
            self.prs[pr] = len(self.prs)
        return self.prs[pr]
    
    def load_contributors(self, src="repo_contributions.txt"):
        dst="contributor_commit_repo.txt"
        with open(os.path.join(self.crawled_data_dir, src), "r", encoding="utf-8") as inf, open(os.path.join(self.processed_data_dir, dst), "w", encoding="utf-8") as outf:
            for line in inf:
                repo, contris = line.strip().split("\t")
                contris = json.loads(contris)
                for c in contris:
                    outf.write("{}\t{}\t{}\n".format(self._get_contributor_id(c[0]), self._get_repo_id(repo), c[1]))

    def load_issues(self, src="repo_issues.txt"):
        issue_belong_to_repo_outf = open(os.path.join(self.processed_data_dir, "issue_belong_to_repo.txt"), "w", encoding="utf-8")
        contributor_propose_issue_outf = open(os.path.join(self.processed_data_dir, "contributor_propose_issue.txt"), "w", encoding="utf-8")

        with open(os.path.join(self.crawled_data_dir, src), "r", encoding="utf-8") as inf:
            for line in inf:
                repo, iss = line.strip().split("\t")
                iss = json.loads(iss)
                repo_id = self._get_repo_id(repo)
                for i in iss:
                    i_name = "{}#{}".format(repo, i["number"])
                    i_id = self._get_issue_id(i_name)
                    committer_id = self._get_contributor_id(i["committer"])
                    issue_belong_to_repo_outf.write("{}\t{}\t{}\n".format(i_id, repo_id, None))
                    contributor_propose_issue_outf.write("{}\t{}\t{}\n".format(committer_id, i_id, None))
    
    def load_prs(self, src="repo_prs.txt", pr_modified_path_file=None):
        modified_code_prs = None
        if pr_modified_path_file and isinstance(pr_modified_path_file, str):
            with open(pr_modified_path_file, "r", encoding="utf-8") as inf:
                modified_code_prs = json.load(inf)
        pr_belong_to_repo_outf = open(os.path.join(self.processed_data_dir, "pr_belong_to_repo.txt"), "w", encoding="utf-8")
        contributor_propose_pr_outf = open(os.path.join(self.processed_data_dir, "contributor_propose_pr.txt"), "w", encoding="utf-8") 
        with open(os.path.join(self.crawled_data_dir, src), "r", encoding="utf-8") as inf:
            for line in inf:
                repo, prs = line.strip().split("\t")
                prs = json.loads(prs)
                repo_id = self._get_repo_id(repo)
                for pr in prs:
                    pr_name = "{}##{}".format(repo, pr["number"])
                    # Filter out those without modifying codes
                    if modified_code_prs and pr_name not in modified_code_prs:
                        continue
                    pr_state = 1
                    if pr["if_merged"]:
                        pr_state = 2
                    elif pr["if_closed"]:
                        pr_state = 0
                    pr_id = self._get_pr_id(pr_name)
                    committer_id = self._get_contributor_id(pr["committer"])

                    pr_belong_to_repo_outf.write("{}\t{}\t{}\n".format(pr_id, repo_id, pr_state))
                    contributor_propose_pr_outf.write("{}\t{}\t{}\n".format(committer_id, pr_id, None))

    def load_stargazers(self, src="repo_stargazers.txt"):
        dst = "contributor_star_repo.txt"
        with open(os.path.join(self.crawled_data_dir, src), "r", encoding="utf-8") as inf, open(os.path.join(self.processed_data_dir, dst), "w", encoding="utf-8") as outf:
            for line in inf:
                repo, stars = line.strip().split("\t")
                stars = json.loads(stars)
                repo_id = self._get_repo_id(repo)
                for s in stars:
                    s_id = self._get_contributor_id(s)
                    outf.write("{}\t{}\t{}\n".format(s_id, repo_id, None))

    def load_watchers(self, src="repo_watchers.txt"):
        dst = "contributor_watch_repo.txt"
        with open(os.path.join(self.crawled_data_dir, src), "r", encoding="utf-8") as inf, open(os.path.join(self.processed_data_dir, dst), "w", encoding="utf-8") as outf:
            for line in inf:
                repo, stars = line.strip().split("\t")
                stars = json.loads(stars)
                repo_id = self._get_repo_id(repo)
                for s in stars:
                    s_id = self._get_contributor_id(s)
                    outf.write("{}\t{}\t{}\n".format(s_id, repo_id, None))

    def load_followings(self):
        user_follow_user = []
        user_followings_src = "user_followings.txt"
        with open(os.path.join(self.crawled_data_dir, user_followings_src), "r", encoding="utf-8") as inf:
            for line in inf:
                usr, followings = line.strip().split("\t")
                usr_id = self._get_contributor_id(usr)
                followings = json.loads(followings)
                for f in followings:
                    user_follow_user.append((usr_id, self._get_contributor_id(f), None))
        
        user_followers_src = "user_followers.txt"
        with open(os.path.join(self.crawled_data_dir, user_followers_src), "r", encoding="utf-8") as inf:
            for line in inf:
                usr, followers = line.strip().split("\t")
                followers = json.loads(followers)
                usr_id = self._get_contributor_id(usr)
                for f in followers:
                    user_follow_user.append((self._get_contributor_id(f), usr_id, None))
        
        dst = "contributor_follow_contributor.txt"
        with open(os.path.join(self.processed_data_dir, dst), "w", encoding="utf-8") as outf:
            user_follow_user_length = len(user_follow_user)
            for item in random.sample(user_follow_user, int(user_follow_user_length * self.follow_sample_percent)):
                outf.write("{}\t{}\t{}\n".format(item[0], item[1], item[2]))
    
    def load_organization(self, src="user_organizations.txt"):
        dst = "contributor_belong_to_org.txt"
        with open(os.path.join(self.crawled_data_dir, src), "r", encoding="utf-8") as inf, open(os.path.join(self.processed_data_dir, dst), "w", encoding="utf-8") as outf:
            for line in inf:
                user, oss = line.strip().split("\t")
                usr_id = self._get_contributor_id(user)
                for o in oss:
                    o_id = self._get_org_id(o)
                    outf.write("{}\t{}\t{}\n".format(usr_id, o_id, None))

    def load_graph(self):
        self.load_contributors()
        self.load_issues()
        self.load_prs(pr_modified_path_file=None)
        self.load_stargazers()
        self.load_watchers()
        self.load_followings()
        self.load_organization()

        with open(os.path.join(self.processed_data_dir, "contributors.json"), "w", encoding="utf-8") as outf:
            json.dump(self.contributors, outf, ensure_ascii=False)

        with open(os.path.join(self.processed_data_dir, "repositories.json"), "w", encoding="utf-8") as outf:
            json.dump(self.repos, outf, ensure_ascii=False)

        with open(os.path.join(self.processed_data_dir, "issues.json"), "w", encoding="utf-8") as outf:
            json.dump(self.issues, outf, ensure_ascii=False)
        
        with open(os.path.join(self.processed_data_dir, "prs.json"), "w", encoding="utf-8") as outf:
            json.dump(self.prs, outf, ensure_ascii=False)
        
        with open(os.path.join(self.processed_data_dir, "organizations.json"), "w", encoding="utf-8") as outf:
            json.dump(self.orgs, outf, ensure_ascii=False)


if __name__ == "__main__":
    crawled_data_dir = "../../GHCrawler/cleaned"
    processed_data_dir = "./full_graph/content"
    follow_sample_percent = 1
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir, exist_ok=True)
    if not os.path.exists(crawled_data_dir):
        print("Source data dir not exists!")
        sys.exit(0)
    cdl = CrawledDataLoader(
        crawled_data_dir=crawled_data_dir, 
        processed_data_dir=processed_data_dir, 
        follow_sample_percent=follow_sample_percent
    )
    cdl.load_graph()
