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

def get_eval_filename(task_name):
    """gets the corresponding output file of an eval task"""
    # remove the part after '::' if it exists
    task_name = task_name.split('::')[0]
    # convert all ":" to "_"
    task_name = task_name.replace(":", "_")
    return f"task-{task_name}"