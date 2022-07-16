import os
import sys
import json
import random
import time
import signal

import requests

def set_timeout(num):
    def wrap(func):
        def handle(signum, frame):
            raise RuntimeError
 
        def to_do(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)  # 设置信号和回调函数
                signal.alarm(num)  # 设置 num 秒的闹钟
                r = func(*args, **kwargs)
                signal.alarm(0)  # 关闭闹钟
                return r
            # 超时返回
            except RuntimeError as e:
                return None
 
        return to_do
 
    return wrap

class Crawler(object):
    def __init__(self, auth_token, log_file):
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": "token {}".format(auth_token)
        }
        self.errlog = open(log_file, "w", encoding="utf-8")
        pass

    def __del__(self):
        self.errlog.close()

    @set_timeout(70)
    def request(self, url, retry=2):
        while retry:
            try:
                response = requests.get(url, headers=self.headers)
            except requests.exceptions.ProxyError as e:
                print("爬取速度过快了，休息一分钟")
                time.sleep(60)
            except requests.exceptions.SSLError as e:
                print("爬取速度过快了，休息一分钟")
                time.sleep(60)
            except requests.exceptions.ConnectionError as e:
                print("爬取速度过快了，休息一分钟")
                time.sleep(60)
            else:
                if response.status_code == 200:
                    return response
                else:
                    return None
            time.sleep(random.randint(1, 3))
            retry -= 1
        return None
    
    def get_owner_and_repo_from_gh_url(self, gh_url:str):
        last_idx = gh_url.rfind("/")
        repo = gh_url[last_idx+1:]
        last_second_idx = gh_url[:last_idx].rfind("/")
        owner = gh_url[last_second_idx+1:last_idx]
        return owner, repo

    def get_gh_url_from_owner_and_repo(self, owner, repo):
        return "https://github.com/{}/{}".format(owner, repo)
    
    def err_handling(self, url):
        # time.sleep(30)
        self.errlog.write(url+"\n")
        pass

class RepositoryCrawler(Crawler):
    def __init__(
        self, 
        auth_token="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", 
        log_file="./log/respository_err.log"):
        super(RepositoryCrawler, self).__init__(auth_token, log_file)

    def crawl_with_github_url(self, gh_url):
        owner, repo = self.get_owner_and_repo_from_gh_url(gh_url)
        return self.crawl(owner, repo)

    def crawl_with_target_url(self, url):
        print("Now crawling URL: {}".format(url))

        response = self.request(url)
        if not response:
            self.err_handling(url)
            return {}
        return response.json()
    
    def crawl(self, owner, repo):
        url = "https://api.github.com/repos/{}/{}".format(owner, repo)
        return self.crawl_with_target_url(url)
        

class ContributorCrawler(Crawler):
    def __init__(
        self, 
        per_page,
        auth_token="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", 
        log_file="./log/contributor_err.log"):
        self.per_page = per_page
        super(ContributorCrawler, self).__init__(auth_token, log_file)

    def crawl_with_github_url(self, gh_url):
        owner, repo = self.get_owner_and_repo_from_gh_url(gh_url)
        return self.crawl(owner, repo)
    
    def crawl_with_target_url(self, url):
        print("Now crawling URL: {}".format(url))
        results, should_continue = list(), True
        response = self.request(url)
        if not response or not response.json():
            self.err_handling(url)
            return results, False

        for item in response.json():
            if item["contributions"] < contributors_bound_limit:
                should_continue = False
                break
            else:
                results.append([item["login"], item["contributions"]])
        return response, should_continue
    
    def crawl(self, owner, repo, contributors_bound_limit=10):
        page_no = 0
        should_continue = True
        results = list()
        while should_continue:
            page_no += 1
            url = "https://api.github.com/repos/{}/{}/contributors?per_page={}&page={}".format(owner, repo, self.per_page, page_no)

            page_result, should_continue = self.crawl_with_target_url(url)
            results.extend(page_result)
        return results

class ContentCrawler(Crawler):
    def __init__(
        self, 
        suffixs, 
        ignored_prefixs,
        per_page,
        cache_url,
        auth_token="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",  
        log_file="./log/content_err.log"):
        super(ContentCrawler, self).__init__(auth_token, log_file)
        self.suffixs = suffixs
        self.ignored_prefixs = ignored_prefixs
        self.per_page = per_page
        self.crawled_urls = set()
        with open(cache_url, "r", encoding="utf-8") as inf:
            for line in inf:
                self.crawled_urls.add(line.strip())
        self.cache_url_file = open(cache_url, "a+", encoding="utf-8")
    
    def __del__(self):
        self.cache_url_file.close()
    
    def crawl(self, owner, repo, dst_out_file, path=None):
        if not path:
            url = "https://api.github.com/repos/{}/{}/contents?ref=main&per_page={}".format(owner, repo, self.per_page)
        else:
            url = "https://api.github.com/repos/{}/{}/contents/{}?ref=main&per_page={}".format(owner, repo, path, self.per_page)
        if url in self.crawled_urls:
            return
        
        print("Now crawling URL: {}".format(url))
        
        response = self.request(url)
        if not response:
            self.err_handling(url)

        response = response.json()
        # file
        if isinstance(response, dict):
            self.cache_url_file.write(url+"\n")
            dst_out_file.write(json.dumps(response, ensure_ascii=False)+"\n")
        # dir
        elif isinstance(response, list):
            for item in response:
                if item["type"] == "file":
                    need_crawl = False
                    for s in self.suffixs:
                        if item["name"].endswith(s):
                            need_crawl = True
                            break
                    if need_crawl:
                        self.crawl(owner, repo, dst_out_file, path=item["path"])
                elif item["type"] == "dir":
                    # print(path, item)
                    need_crawl = True
                    for s in self.ignored_prefixs:
                        if item["path"].startswith(s):
                            need_crawl = False
                            break
                    if need_crawl:
                        self.crawl(owner, repo, dst_out_file, path=item["path"])
