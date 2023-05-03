import os
import json
from tqdm import tqdm
from multiprocessing import Process

from tree_sitter import Language, Parser

SRC_DIR = "../../../DTEM_REPOS"
suffix_to_lang = {
    "py": "python",
    "java": "java",
    "php": "php",
    "rb": "ruby",
    "go": "go",
    "js": "javascript"
}

class MethodExtractor(object):
    def __init__(self, lang) -> None:
        LANG = Language(f'../parser/build/{lang}.so', f'{lang}')
        self.parser = Parser()
        self.parser.set_language(LANG)

    def get_method(self, node, method_list):
        if node.type in [
            "method_declaration",                              # java
            "function", "arrow_function", "method_definition", # javascript
            "function_definition",                             # python
            "method", "singleton_method",                      # ruby
            "function_declaration", "method_declaration",      # go
            "function_definition",                             # php
        ]:
            method_list.append(node)
            return
        for child in node.children:
            self.get_method(child, method_list)
        return  

    def extract_method(self, code_filepath):
        encoding = "utf-8"
        with open(code_filepath, "r", encoding=encoding) as inf:
            try:
                code_src = inf.read()
            except Exception as e:
                encoding="gbk"
                try:
                    with open(code_filepath, "r", encoding=encoding) as inf:
                        code_src = inf.read()
                except:
                    print("Unexpected encoding file passed.")
                    code_src = ""
        
        byted = bytes(code_src, "utf-8")
        tree = self.parser.parse(byted)

        root = tree.root_node
        methods = []
        try:
            self.get_method(root, methods)
        except:
            methods = []
            print("Extracting Method error. Pass this method.")
        method_strs = []
        for method in methods:
            method_strs.append(byted[method.start_byte: method.end_byte].decode("utf-8"))
        return method_strs

def get_lang(filename:str):
    suffix = filename.split(".")[-1]
    if suffix in suffix_to_lang:
        return True, suffix_to_lang[suffix]
    else:
        return False, None

def extract_code_snippets(project_subdirs, finished_projects, process_no):
    python_outf = open(f"./codes/python_{process_no}.jsonl", "a+", encoding="utf-8")
    java_outf = open(f"./codes/java_{process_no}.jsonl", "a+", encoding="utf-8")
    javascript_outf = open(f"./codes/javascript_{process_no}.jsonl", "a+", encoding="utf-8")
    php_outf = open(f"./codes/php_{process_no}.jsonl", "a+", encoding="utf-8")
    ruby_outf = open(f"./codes/ruby_{process_no}.jsonl", "a+", encoding="utf-8")
    go_outf = open(f"./codes/go_{process_no}.jsonl", "a+", encoding="utf-8")

    for project_subdir in project_subdirs:
        project = project_subdir.replace("#", "/")
        for filename in os.listdir(os.path.join(SRC_DIR, project_subdir)):
            if filename in finished_projects.get(project, []):
                continue
            is_code, lang = get_lang(filename)
            if not is_code:
                continue

            # print("Handling", os.path.join(SRC_DIR, project_subdir, filename),)
            
            code_filepath = os.path.join(SRC_DIR, project_subdir, filename)
            method_strings = []
            outf = None

            if lang == "python":
                method_strings = python_extractor.extract_method(code_filepath)
                outf = python_outf
                pass
            elif lang == "javascript":
                method_strings = javascript_extractor.extract_method(code_filepath)
                outf = javascript_outf
                pass
            elif lang == "java":
                method_strings = javascript_extractor.extract_method(code_filepath)
                outf = java_outf
                pass
            elif lang == "ruby":
                method_strings = ruby_extractor.extract_method(code_filepath)
                outf = ruby_outf
                pass
            elif lang == "go":
                method_strings = go_extractor.extract_method(code_filepath)
                outf = go_outf
                pass
            elif lang == "php":
                method_strings = php_extractor.extract_method(code_filepath)
                outf = php_outf
                pass
            outf.write(
                json.dumps({
                    "project": project,
                    "path": filename.replace("\\", "/"),
                    "method_strings": method_strings,
                    "lang": lang
                }, 
                ensure_ascii=False)+"\n"
            )

def multi_process_execution():
    finished_projects = dict()
    if os.path.exists("./finished_repos.json"):
        with open("./finished_repos.json", "r", encoding="utf-8") as inf:
            finished_projects = json.load(inf)
    projects = []
    for filename in os.listdir(SRC_DIR):
        if "#" not in filename:
            continue
        projects.append(filename)
    
    processes = []
    for i in range(8):
        processes.append(Process(target=extract_code_snippets, args=(projects[i * 10000: (i+1) * 10000], finished_projects, i)))
    [p.start() for p in processes]  # 开启进程
    [p.join() for p in processes]   # 等待进程依次结束

def update_finished_projects():
    finished_projects = dict()
    for filename in os.listdir("./codes"):
        with open(os.path.join("./codes", filename), "r", encoding="utf-8") as inf:
            for line in inf:
                obj = json.loads(line.strip())
                project = obj["project"]
                filen = obj["path"].replace("/", "\\")
                if project not in finished_projects:
                    finished_projects[project] = [filen]
                else:
                    finished_projects[project].append(filen)
    with open("./finished_repos.json", "w", encoding="utf-8") as outf:
        json.dump(finished_projects, outf, ensure_ascii=False)

if __name__ == "__main__":
    python_extractor = MethodExtractor("python")
    go_extractor = MethodExtractor("go")
    java_extractor = MethodExtractor("java")
    javascript_extractor = MethodExtractor("javascript")
    ruby_extractor = MethodExtractor("ruby")
    php_extractor = MethodExtractor("php")
    multi_process_execution()
    
    update_finished_projects()