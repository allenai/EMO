import json
import os


def find_file(directory, substring):
    """Finds all files in directory that contain substring in their filename."""
    found_arr = []
    for root, _, files in os.walk(directory):
        for file in files:
            if substring in file:
                found_arr += [os.path.join(root, file)]
    return found_arr


def load_jsonl_file(file_path):
    """Loads a jsonl file and returns a list of json objects."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def find_task_substring(task_name):
    """finds a substring that can use find_file to locate files for a given task name"""
    if "mc" in task_name:
        return task_name.split("::")[0]
    if "rc" in task_name:
        return task_name.split(":rc")[0]
    raise NotImplementedError(f"Task {task_name} not implemented in find_task_substring")