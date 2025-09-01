import os
import pandas as pd
from pathlib import Path
import sys

def getFileLocation(file_name):
    if os.path.exists(file_name):
        return file_name
    
    dirs_to_check = []
    script_dir = Path(__file__) #already checked
    root_dir = script_dir.parent #same level as datapipelinefolder

    dirs_to_check.append(root_dir)

    root_parent_dir = root_dir.parent
    dirs_to_check.append(root_parent_dir) #check if its main project dir

    project_dir = root_parent_dir.parent
    dirs_to_check.append(project_dir)

    dirs_to_check.append(os.path.join(project_dir,'backTestingData')) #check if its in the shared data folder

    for directory in dirs_to_check:
        current_path = os.path.join(directory,file_name)
        print(f'checking > {current_path}')
        if os.path.exists(current_path):
            return current_path


def loadData(path):
    files = os.listdir(path)
    read_dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(path,file))
        name,extension = os.path.splitext(file)
        df['Ticker'] = name
        read_dfs.append(df)

    return read_dfs


def run(data_folder):
    path = getFileLocation(data_folder)

    read_dfs = loadData(path)

    return read_dfs
