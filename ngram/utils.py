import os
import shutil
import re
import json
import subprocess
from tqdm import tqdm
import random
import datetime
import pytz
import javalang
from javalang.tokenizer import LexerError
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from .repo_names import *


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

    with tqdm(total=len(repo_name_list)) as progress_bar:
        # for one_repo_name in tqdm(repo_name_list):
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
def split_the_files_in_data(input_dir, output_dir, train_num_list, test_num):
    # Define the split ratios
    train_ratio = 0.9  # for training
    # val_ratio = 0.0  # for validation
    test_ratio = 0.1  # for testing

    # train_target_list = [16000]#[125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    test_target = test_num

    # Create directories for train, val, and test sets
    # train_dir = os.path.join(output_dir, "train")
    train_dir_list = [os.path.join(output_dir, f"train_{item}") for item in train_num_list]
    # val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    # os.makedirs(train_dir, exist_ok=True)
    for one_train_dir in train_dir_list:
        os.makedirs(one_train_dir, exist_ok=True)
    # os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all .java files from the data directory
    java_files = [f for f in os.listdir(input_dir) if f.endswith('.java')]

    # Shuffle the list of files to ensure random selection
    random.shuffle(java_files)

    # num_of_used_files = 1000
    num_of_used_files = len(java_files)
    print(f"num_of_used_files = {num_of_used_files}")
    # Split the files based on the defined ratios
    train_split = int(train_ratio * num_of_used_files )
    val_split = int(train_ratio * num_of_used_files )

    train_files = java_files[:train_split]
    # val_files = java_files[train_split:val_split]
    test_files = java_files[val_split:val_split + test_target]

    for i, one_train_size in enumerate(train_num_list):
        random.shuffle(train_files)
        specific_size_train_files = train_files[:one_train_size]
        for file_name in specific_size_train_files:
            src_path = os.path.join(input_dir, file_name)
            dest_path = os.path.join(train_dir_list[i], file_name)
            shutil.copy(src_path, dest_path)
        print(f"File number in train_{one_train_size} is:", count_files_num_in_folder(train_dir_list[i]))


    # Move the files to their respective directories
    # for file_name in train_files:
    #     src_path = os.path.join(data_dir, file_name)
    #     dest_path = os.path.join(train_dir, file_name)
    #     shutil.copy(src_path, dest_path)
    # print("the file number in train_file is:",count_files_num_in_folder(train_dir))
    #
    # for file_name in val_files:
    #     src_path = os.path.join(data_dir, file_name)
    #     dest_path = os.path.join(val_dir, file_name)
    #     shutil.copy(src_path, dest_path)
    # print("the file number in train_file is:",count_files_num_in_folder(val_dir)  )

    for file_name in test_files:
        src_path = os.path.join(input_dir, file_name)
        dest_path = os.path.join(test_dir, file_name)
        shutil.copy(src_path, dest_path)
    print("File number in test is:", count_files_num_in_folder(test_dir))

    print(f"Files have been split and moved to {output_dir}.")

# YL, use the tokenizer "javalang" to tokenize the strings in the data_sets, and write the results into the folder named "train_set_tokenizer"

# Function to tokenize Java code
def tokenize_java_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    try:
        tokens = list(javalang.tokenizer.tokenize(code))
        return tokens
    except LexerError as e:
        # print(f"LexerError in file {file_path}: {e}")
        return None


# Function to write tokenized output to a new file
def write_tokenized_output(output_path, tokens):
    with open(output_path, 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(str(token) + '\n')

def tokenize_data_sets(input_dir, output_dir):
    # # Define directories for input and output
    # input_dirs = {
    #     # "train_set": "data_sets/train_set",
    #     # "val_set": "data_sets/val_set",
    #     # "test_set": "data_sets/test_set",
    #     "all_set": "data",
    # }
    #
    # output_dirs = {
    #     # "train_set_tokenized": "data_sets/train_set_tokenized",
    #     # "val_set_tokenized": "data_sets/val_set_tokenized",
    #     # "test_set_tokenized": "data_sets/test_set_tokenized",
    #     "all_set_tokenized": "data_token",
    # }

    # Create output directories if they don't exist
    # for output_dir in output_dirs.values():
    #     os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Tokenize and save for each set (train/test)
    # for set_name, input_dir in input_dirs.items():
    #     output_dir = output_dirs[f"{set_name}_tokenized"]
    for java_file in tqdm(os.listdir(input_dir)):
        if java_file.endswith(".java"):
            input_file_path = os.path.join(input_dir, java_file)
            output_file_path = os.path.join(output_dir, java_file)

            # Tokenize the Java file
            try:
                tokens = tokenize_java_file(input_file_path)
                if tokens:  # If tokenization was successful
                    # Write tokenized content to a new file
                    write_tokenized_output(output_file_path, tokens)
            except Exception as e:
                print(e)
                tokens = None


    file_input_count = count_files_num_in_folder(input_dir)
    file_output_count = count_files_num_in_folder(output_dir)

    print(f"Number of files in the folder '{input_dir}': {file_input_count}")
    print(f"Number of files in the folder '{output_dir}': {file_output_count}")

    print("Tokenization completed for all sets.")


# count how many files are there in one folder: eg: folder_path = 'data_sets/test_set_tokenized'
def count_files_num_in_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return len(files)


def get_timestring(time_string="%Y%m%d_%H%M%S_%f"):
    est = pytz.timezone('America/New_York')
    utc_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    est_now = utc_now.astimezone(est)
    return est_now.strftime(time_string)

# def calculate_metrics(ground_truth_lists, predicted_lists):
#     """
#     Calculate Precision, Recall, F1 Score, and Accuracy between the ground truth and predicted tokens.
#     :param ground_truth_lists: A list of ground truth token lists (val_token_lists).
#     :param predicted_lists: A list of predicted token lists.
#     :return: Precision, Recall, F1 Score, and Accuracy.
#     """
#
#     # Flatten the lists to calculate metrics across all tokens
#     ground_truth_flat = [token for sublist in ground_truth_lists for token in sublist]
#     predicted_flat = [token for sublist in predicted_lists for token in sublist]
#
#     # Ensure both lists are of the same length
#     min_length = min(len(ground_truth_flat), len(predicted_flat))
#     ground_truth_flat = ground_truth_flat[:min_length]
#     predicted_flat = predicted_flat[:min_length]
#
#     # Initialize counts
#     tp=0
#
#     # not correct yet
#     for gt, pred in zip(ground_truth_flat, predicted_flat):
#         if gt == pred:
#             tp+=1
#
#     # Calculate Precision, Recall, F1, and Accuracy
#     precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
#     recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
#     f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#     accuracy = (true_positive + true_negative) / len(ground_truth_flat) if len(ground_truth_flat) > 0 else 0
#
#     return precision, recall, f1, accuracy


if __name__ == "__main__":
    # repo_url = "https://github.com/00-evan/pixel-dungeon-gradle"
    # destination_folder = "renamed_java_files"
    # extract_and_rename_java_files(repo_url, destination_folder)
    # num_c, num_f = process_java_file("test/41.java", "test/41.txt")
    # print(num_c, num_f)
    # sum_class: 126269, sum_func: 1136926
    # Index file saved to index.json
    # build_dataset_files(repo_names, 70, 100)

    #remember, everytime run the split_the_files_in_data(), delete the data_sets, otherwise it will generate many!!!!!!!!
    split_the_files_in_data("data_token", "data_processed")
    #tokenize the train/val/test_set and save the results
    # tokenize_data_sets()


    # groundtruth_list = [ ['public', 'class', 'ErrorCodeTest', 'extends', 'TestCase', '{', 'public', 'void', 'testErrorCodes', '(', ')', 'throws', 'Exception', '{', 'HashMap', '<', 'Byte', ',', 'String', '>', 'errMap', '=', 'new', 'HashMap', '<', 'Byte', ',', 'String', '>', '(', ')', ';', 'OperationFactory', 'opFact', '=', 'new', 'BinaryOperationFactory', '(', ')', ';', 'errMap', '.', 'put', '(', 'new', 'Byte', '(', '(', 'byte', ')', '0x01', ')', ',', 'NOT', ')', ';', 'errMap', '.', 'put', '(', 'new', 'Byte', '(', '(', 'byte', ')', '0x02', ')', ',', 'EXISTS', ')', ';', 'errMap', '.', 'put', '(', 'new', 'Byte', '(', '(', 'byte', ')', '0x03', ')', ',', '2BIG', ')', ';', 'errMap', '.', 'put', '(', 'new', 'Byte', '(', '(', 'byte', ')', '0x04', ')', ',', 'INVAL', ')', ';', 'errMap', '.', 'put', '(', 'new', 'Byte', '(', '(', 'byte', ')', '0x05', ')', ',', 'NOT', ')', ';', 'errMap', '.', 'put', '(', 'new', 'Byte', '(', '(', 'byte', ')', '0x06', ')', ',', 'DELTA', ')', ';', 'errMap', '.', 'put', '(', 'new', 'Byte', '(', '(', 'byte', ')', '0x07', ')', ',', 'NOT', ')', ';', 'errMap', '.', 'put', '(', 'new', 'Byte', '(', '(', 'byte', ')', '0x81', ')', ',', 'UNKNOWN', ')', ';', 'errMap', '.', 'put', '(', 'new', 'Byte', '(', '(', 'byte', ')', '0x82', ')', ',', 'NO', ')', ';', 'errMap', '.', 'put', '(', 'new', 'Byte', '(', '(', 'byte', ')', '0x83', ')', ',', 'NOT', ')', ';', 'errMap', '.', 'put', '(', 'new', 'Byte', '(', '(', 'byte', ')', '0x84', ')', ',', 'INTERNAL', ')', ';', 'errMap', '.', 'put', '(', 'new', 'Byte', '(', '(', 'byte', ')', '0x85', ')', ',', 'BUSY', ')', ';', 'errMap', '.', 'put', '(', 'new', 'Byte', '(', '(', 'byte', ')', '0x86', ')', ',', 'TEMP', ')', ';', 'int', 'opaque', '=', '0', ';', 'for', '(', 'final', 'Entry', '<', 'Byte', ',', 'String', '>', 'err', ':', 'errMap', '.', 'entrySet', '(', ')', ')', '{', 'byte', '[', ']', 'b', '=', 'new', 'byte', '[', '24', '+', 'err', '.', 'getValue', '(', ')', '.', 'length', '(', ')', ']', ';', 'b', '[', '0', ']', '=', '(', 'byte', ')', '0x81', ';', 'b', '[', '7', ']', '=', 'err', '.', 'getKey', '(', ')', ';', 'b', '[', '11', ']', '=', '(', 'byte', ')', 'err', '.', 'getValue', '(', ')', '.', 'length', '(', ')', ';', 'b', '[', '15', ']', '=', '(', 'byte', ')', '++', 'opaque', ';', 'System', '.', 'arraycopy', '(', 'err', '.', 'getValue', '(', ')', '.', 'getBytes', '(', ')', ',', '0', ',', 'b', ',', '24', ',', 'err', '.', 'getValue', '(', ')', '.', 'length', '(', ')', ')', ';', 'GetOperation', 'op', '=', 'opFact', '.', 'get', '(', 'key', ',', 'new', 'GetOperation', '.', 'Callback', '(', ')', '{', 'public', 'void', 'receivedStatus', '(', 'OperationStatus', 's', ')', '{', 'assert', '!', 's', '.', 'isSuccess', '(', ')', ';', 'assert', 'err', '.', 'getValue', '(', ')', '.', 'equals', '(', 's', '.', 'getMessage', '(', ')', ')', ';', '}', 'public', 'void', 'gotData', '(', 'String', 'k', ',', 'int', 'flags', ',', 'byte', '[', ']', 'data', ')', '{', '}', 'public', 'void', 'complete', '(', ')', '{', '}', '}', ')', ';', 'ByteBuffer', 'bb', '=', 'ByteBuffer', '.', 'wrap', '(', 'b', ')', ';', 'bb', '.', 'flip', '(', ')', ';', 'op', '.', 'readFromBuffer', '(', 'bb', ')', ';', '}', '}', '}'] ]
    # predict_list = [ ['public', 'class', 'ErrorCodeTest', 'extends'] ]
    # precision, recall, f1, accuracy = calculate_metrics(groundtruth_list, predict_list)
    #
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")
    # print(f"Accuracy: {accuracy}")
