import os
import sys
import shutil

class CodeFileExtractor(object):
    def __init__(self, 
                 suffixes=[".py", ".ipynb", ".java", ".js", ".php", ".rb", ".go"]):
        self.suffixes = suffixes
    
    def get_owner_and_repo_name(self, project_dir):
        owner, repo = project_dir.split("#")
        return owner, repo
    
    def extract_code_files_in_project(self, project_dir, out_dir):
        def check_match_suffix(filename:str):
            if not self.suffixes:
                return True
            for s in self.suffixes:
                if filename.endswith(s):
                    return True
            return False
        
        for filenanme in os.listdir(project_dir):
            out_dir = os.path.join(out_dir, project_dir)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            filepath = os.path.join(project_dir, filenanme)
            if os.path.isdir(filepath):
                self.extract_code_files_in_project(filepath, out_dir)
            elif os.path.isfile(filepath):
                if check_match_suffix(fileanme):
                    shutil.copy(filepath, os.path.join(out_dir, filename))

if __name__ == "__main__":
    pass