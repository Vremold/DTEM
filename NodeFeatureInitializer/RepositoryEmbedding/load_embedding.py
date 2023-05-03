import os
import re
import sys
import json

from bs4 import BeautifulSoup
from markdown import markdown

readme_dir = "../../GHCrawler/readme"
readme_fnames = set(os.listdir(readme_dir))

def markdown_to_text_and_code(markdown_string):
    """ Converts a markdown string to plaintext """
    # extract code snippets and remove them from raw texts
    codes = re.findall(r"`+([^`]+?)`+", markdown_string)
    markdown_string = re.sub(r"`+[^`]+?`+", "", markdown_string)

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string, extensions=['markdown.extensions.tables', 'markdown.extensions.md_in_html'])

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(string=True))

    return text, codes

def extract_code_text_from_raw(text:str):
    text, codes = markdown_to_text_and_code(text)
    text = re.sub(r"[\t\r\n ]+", " ", text).strip()
    return text, codes

def extract_text(src_file="", dst_file=""):
    with open(src_file, "r", encoding="utf-8") as inf, open(dst_file, "w", encoding="utf-8") as outf:
        for line in inf:
            obj = json.loads(line)
            repo = obj["full_name"].lower()
            fname = repo.replace("/", "#") + ".md"
            readme_text = ""
            if fname in readme_fnames:
                # print("reading readme file: {}".format(fname))
                with open(os.path.join(readme_dir, fname), "r", encoding="utf-8") as f:
                    readme_text = f.read()
                readme_text, _ = extract_code_text_from_raw(readme_text)
            text = obj["description"]
            if text is None:
                text = ""
            text, _ = extract_code_text_from_raw(text)
            text = text + readme_text
            if not text:
                continue
            outf.write(
                json.dumps({
                    "project": repo,
                    "text": text,
                }, ensure_ascii=False)+"\n"
            )
    pass

if __name__ == "__main__":
    extract_text(src_file="../../GHCrawler/cleaned/repo_statistics.txt", dst_file="./repo_descriptions.txt")