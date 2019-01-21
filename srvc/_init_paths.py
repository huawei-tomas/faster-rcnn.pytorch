import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

def add_libpath():
    srvc_dirname = osp.dirname(__file__)
    fstrrcnn_dirname = osp.dirname(srvc_dirname)
    lib_path = osp.join(fstrrcnn_dirname, "lib")
    add_path(fstrrcnn_dirname)
    add_path(lib_path)

add_libpath()
