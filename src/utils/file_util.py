import os

def get_project_root(start_path=None):
    if start_path is None:
        start_path = os.path.abspath(__file__)

    # 向上遍历目录，直到找到 '.git' 目录
    while True:
        if os.path.isdir(os.path.join(start_path, '.git')):
            return start_path

        parent_path = os.path.dirname(start_path)
        if parent_path == start_path:
            # 如果已经到达了文件系统的根目录，则停止遍历
            raise FileNotFoundError("Could not find project root")

        start_path = parent_path
def get_file_path(filename):
    path = os.path.join(get_project_root(), filename)
    return os.path.normpath(path)