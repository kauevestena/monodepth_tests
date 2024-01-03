from lib import *
from shutil import rmtree

for path in [zerodepth_tests_folderpath, packnet_tests_folderpath, zerodepth_pcloud_tests_folderpath, packnet_pcloud_tests_folderpath]:
    if os.path.exists(path):
        rmtree(path)

    create_dir(path)