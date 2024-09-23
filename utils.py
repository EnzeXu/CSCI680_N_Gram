import os
import shutil
import re
import json
import subprocess
import tqdm
import random
import javalang

from repo_names import *


def extract_and_rename_java_files(repo_url, destination_folder, index, index_dic, progress_bar=None):
    tmp_folder = './tmp'
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(tmp_folder, repo_name)

    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    # print(f"Cloning the repository {repo_url} into {tmp_folder}...")
    # subprocess.run(["git", "clone", repo_url, repo_path])
    try:
        subprocess.run(
            ["git", "clone", repo_url, repo_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10
        )
    except subprocess.TimeoutExpired:
        print(f"Error: Cloning the repository {repo_url} took too long (more than 10 seconds).")
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        return index

    java_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for java_file in java_files:
        new_filename = f"{index:08d}.java"
        destination_path = os.path.join(destination_folder, new_filename)
        # shutil.copy(java_file, destination_path)
        try:
            num_of_class, num_of_func = process_java_file(java_file, destination_path)
        except Exception as e:
            print(f"Errors in file \"{java_file}\"", e)
            num_of_class, num_of_func = 0, 0
        if num_of_class > 0:
            index_dic["class"][str(index)] = num_of_class
            index_dic["func"][str(index)] = num_of_func
            index_dic["source"][str(index)] = java_file[6:]
            if not progress_bar:
                print(f"index: {index:08d} class: {num_of_class:02d} func: {num_of_func:03d} filename: {java_file[6:]}")
            else:
                progress_bar.set_description(f"index: {index:08d} class: {num_of_class:02d} func: {num_of_func:03d} filename: {java_file[6:]}")
                print(f"index: {index:08d} class: {num_of_class:02d} func: {num_of_func:03d} filename: {java_file[6:]}")
            index += 1


    shutil.rmtree(repo_path)
    return index


def process_java_file(java_file, destination_path):
    num_of_class = 0
    num_of_func = 0

    with open(java_file, "r") as f:
        content = f.read()

    content = re.sub(r'//.*?(\n|$)', '', content)  # remove single-line comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)  # remove multi-line comments

    class_match = re.search(r'public class\s+\w+', content)

    if class_match:
        # get the position of the class and find the matching closing brace "}"
        class_start = class_match.start()
        class_end = find_class_end(content, class_start)
        content = content[class_start:class_end]
        num_of_class = 1
    else:
        # if no "public class" is found, clear the content
        content = ""

    # symbols = [",", ".", "}", "{", ";", "(", ")", "[", "]", "+", "-", "*", "/", "<", ">", "=", "@", "#", "~", "&"]
    symbols = [".", ",", ";", "@", "#", ":", "(", ")", "[", "]", "{", "}", ">", "<"]
    # for symbol in symbols:
    #     content = content.replace(symbol, f' {symbol} ')

    content = re.sub(r'\s+', ' ', content).strip()

    # count number of functions
    function_pattern = r'\b(?:public|private|protected)?\s*(?:static)?\s*\w+\s+\w+\s*\([^)]*\)\s*(?:throws\s+\w+(?:,\s*\w+)*)?\s*\{'
    num_of_func = len(re.findall(function_pattern, content))

    with open(destination_path, 'w') as f:
        f.write(content)

    return num_of_class, num_of_func


def find_class_end(content, class_start):
    brace_count = 0
    for i in range(class_start, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return i + 1
    return len(content)


def build_dataset_files(raw_repo_names, start, end):
    repo_name_list = raw_repo_names.split()
    repo_name_list = repo_name_list[start:end]
    index_save_path = "index.json"

    if os.path.exists(index_save_path):
        with open(index_save_path, "r") as f:
            index_dic = json.load(f)
        index = index_dic["index"]
        print(f"Loaded {index_save_path}: index = {index}")
    else:
        index = 0
        index_dic = dict()
        index_dic["class"] = dict()
        index_dic["func"] = dict()
        index_dic["source"] = dict()
        index_dic["repo_list"] = []

    with tqdm.tqdm(total=len(repo_name_list)) as progress_bar:
        # for one_repo_name in tqdm.tqdm(repo_name_list):
        for one_repo_name in repo_name_list:
            if one_repo_name in index_dic["repo_list"]:
                print(f"skip repo {one_repo_name}")
            else:
                repo_url = f"https://github.com/{one_repo_name}"
                index = extract_and_rename_java_files(repo_url, "data/", index, index_dic, progress_bar)
            progress_bar.update(1)
            index_dic["repo_list"].append(one_repo_name)
    index_dic["index"] = index

    sum_class = sum([index_dic["class"][str(i)] for i in range(index) if str(i) in index_dic["class"]])
    sum_func = sum([index_dic["func"][str(i)] for i in range(index) if str(i) in index_dic["func"]])

    index_dic["sum_class"] = sum_class
    index_dic["sum_func"] = sum_func

    print(f"sum_class: {sum_class}, sum_func: {sum_func}")


    with open(index_save_path, "w") as f:
        json.dump(index_dic, f, indent=4)

    print(f"Index file saved to {index_save_path}")

#YL split the files in folder "data" into train, validation and test sets, and write the generated sets into the folder "data_set"
def split_the_files_in_data(source_dir, target_dir):
    # Set the paths
    data_dir = source_dir  # Your original data directory containing the .java files
    output_dir = target_dir  # Folder where train, val, test sets will be stored

    # Define the split ratios
    train_ratio = 0.8  # 80% for training
    val_ratio = 0.1  # 10% for validation
    test_ratio = 0.1  # 10% for testing

    # Create directories for train, val, and test sets
    train_dir = os.path.join(output_dir, "train_set")
    val_dir = os.path.join(output_dir, "val_set")
    test_dir = os.path.join(output_dir, "test_set")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all .java files from the data directory
    java_files = [f for f in os.listdir(data_dir) if f.endswith('.java')]

    # Shuffle the list of files to ensure random selection
    random.shuffle(java_files)

    num_of_used_files = 1000
    # Split the files based on the defined ratios
    train_split = int(train_ratio * num_of_used_files )
    val_split = int( ( train_ratio + val_ratio ) * num_of_used_files )

    train_files = java_files[:train_split]
    val_files = java_files[train_split:val_split]
    test_files = java_files[val_split:num_of_used_files]

    # Move the files to their respective directories
    for file_name in train_files:
        src_path = os.path.join(data_dir, file_name)
        dest_path = os.path.join(train_dir, file_name)
        shutil.copy(src_path, dest_path)
    print("the file number in train_file is:",len(train_files) )

    for file_name in val_files:
        src_path = os.path.join(data_dir, file_name)
        dest_path = os.path.join(val_dir, file_name)
        shutil.copy(src_path, dest_path)
    print("the file number in validation_file is:",len(val_files) )


    for file_name in test_files:
        src_path = os.path.join(data_dir, file_name)
        dest_path = os.path.join(test_dir, file_name)
        shutil.copy(src_path, dest_path)
    print("the file number in test_file is:",len(test_files) )


    print(f"Files have been split and moved to {output_dir}.")

# YL, use the tokenizer "javalang" to tokenize the strings in the data_sets, and write the results into the folder named "train_set_tokenizer"

# Function to tokenize Java code
def tokenize_java_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    tokens = list(javalang.tokenizer.tokenize(code))
    return tokens


# Function to write tokenized output to a new file
def write_tokenized_output(output_path, tokens):
    with open(output_path, 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(str(token) + '\n')

def tokenize_data_sets():
    # Define directories for input and output
    input_dirs = {
        "train_set": "data_sets/train_set",
        "val_set": "data_sets/val_set",
        "test_set": "data_sets/test_set"
    }

    output_dirs = {
        "train_set_tokenized": "data_sets/train_set_tokenized",
        "val_set_tokenized": "data_sets/val_set_tokenized",
        "test_set_tokenized": "data_sets/test_set_tokenized"
    }

    # Create output directories if they don't exist
    for output_dir in output_dirs.values():
        os.makedirs(output_dir, exist_ok=True)


    # Tokenize and save for each set (train/val/test)
    for set_name, input_dir in input_dirs.items():
        output_dir = output_dirs[f"{set_name}_tokenized"]
        for java_file in os.listdir(input_dir):
            if java_file.endswith(".java"):
                input_file_path = os.path.join(input_dir, java_file)
                output_file_path = os.path.join(output_dir, java_file)

                # Tokenize the Java file
                tokens = tokenize_java_file(input_file_path)
                # print( "tokens:", tokens )
                # Write tokenized content to a new file
                write_tokenized_output(output_file_path, tokens)

    print("Tokenization completed for all sets.")



if __name__ == "__main__":
    # repo_url = "https://github.com/00-evan/pixel-dungeon-gradle"
    # destination_folder = "renamed_java_files"
    # extract_and_rename_java_files(repo_url, destination_folder)
    # num_c, num_f = process_java_file("test/41.java", "test/41.txt")
    # print(num_c, num_f)
    # sum_class: 126269, sum_func: 1136926
    # Index file saved to index.json
    # build_dataset_files(repo_names, 0, 17)
    # split_the_files_in_data("data", "data_sets" )

    #tokenize the train/val/test_set and save the results
    tokenize_data_sets()
