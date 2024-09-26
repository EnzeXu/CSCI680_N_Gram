import random
import os
import pickle
import argparse
from collections import defaultdict
from .utils import get_timestring
# from utils import calculate_metrics
from tqdm import tqdm

class N_gram:
    def __init__(self, N, train_num=500):
        self.N = N
        self.train_num = train_num
        self.n_grams = defaultdict(int)   # dictionary to store the N-grams tokens list and their frequencies
        self.n_minus_1_grams = defaultdict(int)    # dictionary to store the (N-1)-grams tokens list and their frequencies
        # self.probabilities = {}   # dictionary to store the probabilities of each N-gram token list
        self.n_minus_1_grams_next_available = defaultdict(list)
        self.token_dataset_train, self.token_dataset_val, self.token_dataset_test = None, None, None
        # self.generate_datasets()
        # self.generate_vocabulary_train()

    def load_tokenized_files(self, folder_path):
        token_lists = []
        # cnt = 0
        # Iterate over all files in the tokenized folder
        file_list = os.listdir(folder_path)
        file_list = sorted([item for item in file_list if item.endswith(".java")])
        for file_name in tqdm(file_list):
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            source_code_per_file = []
            # Read each line (token) and store it in a list for this file
            per_file = [line.strip() for line in lines if line.strip()]
            for line in per_file:
                # print(each_token)
                only_code_token = line.split()[1][1:-1]
                source_code_per_file.append(only_code_token)
            # print(source_code_per_file[-10:])
            token_lists.append(source_code_per_file)
            # cnt+=1;
            # if cnt == 5:
            #     break
        return token_lists


    def generate_datasets(self, train_input_folder, test_input_folder, data_dump_folder="data_dump"):
        if not os.path.exists(data_dump_folder):
            os.makedirs(data_dump_folder)
        train_path = os.path.join(data_dump_folder, f"train_{self.train_num}_{self.N}_gram.pkl")
        test_path = os.path.join(data_dump_folder, f"test_{self.N}_gram.pkl")

        if os.path.exists(train_path):
            print(f"Loading train set from {data_dump_folder}: train_{self.train_num} ...")
            with open(train_path, "rb") as f:
                self.token_dataset_train = pickle.load(f)
        else:
            print(f"Train set not found. Generating ...")
            self.token_dataset_train = self.load_tokenized_files(train_input_folder)
            with open(train_path, "wb") as f:
                pickle.dump(self.token_dataset_train, f)

        if os.path.exists(test_path):
            print(f"Loading test set from {data_dump_folder}: test ...")
            with open(test_path, "rb") as f:
                self.token_dataset_test = pickle.load(f)
        else:
            print(f"Test set not found. Generating ...")
            self.token_dataset_test = self.load_tokenized_files(test_input_folder)
            with open(test_path, "wb") as f:
                pickle.dump(self.token_dataset_test, f)

        print(f"Datasets are ready. train size: {len(self.token_dataset_train)}, test size: {len(self.token_dataset_test)}")

    def generate_vocabulary_train(self, data_dump_folder="data_dump"):
        if not os.path.exists(data_dump_folder):
            os.makedirs(data_dump_folder)
        vocab_path_n_grams_path = os.path.join(data_dump_folder, f"train_{self.train_num}_{self.N}_gram_vocab_n_grams.pkl")
        vocab_path_n_minus_1_grams_path = os.path.join(data_dump_folder, f"train_{self.train_num}_{self.N}_gram_vocab_n_minus_1_grams.pkl")
        vocab_path_n_minus_1_grams_next_available_path = os.path.join(data_dump_folder, f"train_{self.train_num}_{self.N}_gram_vocab_n_minus_1_grams_next_available.pkl")
        if os.path.exists(vocab_path_n_grams_path):
            print(f"Loading vocabularies from {data_dump_folder}: n_grams, n_minus_1_grams and n_minus_1_grams_next_available ...")
            with open(vocab_path_n_grams_path, "rb") as f:
                self.n_grams = pickle.load(f)
            with open(vocab_path_n_minus_1_grams_path, "rb") as f:
                self.n_minus_1_grams = pickle.load(f)
            with open(vocab_path_n_minus_1_grams_next_available_path, "rb") as f:
                self.n_minus_1_grams_next_available = pickle.load(f)
        else:
            print(f"Vocabulary not found. Generating ...")
            self.update_vocabulary_from_token_lists(self.token_dataset_train)
            with open(vocab_path_n_grams_path, "wb") as f:
                pickle.dump(self.n_grams, f)
            with open(vocab_path_n_minus_1_grams_path, "wb") as f:
                pickle.dump(self.n_minus_1_grams, f)
            with open(vocab_path_n_minus_1_grams_next_available_path, "wb") as f:
                pickle.dump(self.n_minus_1_grams_next_available, f)

    def update_vocabulary_from_token_lists(self, token_lists):
        """
        Generate N-gram dictionaries based on tokenized input
        :param token_lists: A list of tokenized sentences (lists of tokens)
        """
        for tokens in tqdm(token_lists):
            # Add start and end markers
            tokens = ['<s>'] * (self.N - 1) + tokens + ['</s>']
            for i in range(len(tokens) - self.N + 1):
                n_gram = tuple(tokens[i:i + self.N])
                n_minus_1_gram = tuple(tokens[i:i + self.N - 1])
                self.n_grams[n_gram] += 1
                self.n_minus_1_grams[n_minus_1_gram] += 1
                if n_gram[-1] not in self.n_minus_1_grams_next_available[n_minus_1_gram]:
                    self.n_minus_1_grams_next_available[n_minus_1_gram].append(n_gram[-1])

        # # Compute probabilities
        # for n_gram in self.n_grams:
        #     n_minus_1_gram = n_gram[:-1]
        #     self.probabilities[n_gram] = self.n_grams[n_gram] / self.n_minus_1_grams[n_minus_1_gram]

    def run_test(self, token_lists, log_path="logs.csv"):
        task_num = 0
        task_success_count = 0
        print(f"Begin: task_success_count = {task_success_count}")
        for tokens in tqdm(token_lists):
            # Add start and end markers
            tokens = ['<s>'] * (self.N - 1) + tokens + ['</s>']
            task_list = []
            for i in range(len(tokens) - self.N + 1):
                task_list.append(tokens[i:i + self.N])

            for one_task in task_list:
                given_tokens = one_task[:-1]
                truth_token = one_task[-1]
                pred_token = self.predict_next_token(given_tokens)
                task_num += 1
                task_success_count += int(truth_token == pred_token)
        precision = task_success_count / task_num
        print(f"N: {self.N}, train_num: {self.train_num} | precision on the test set: {task_success_count} / {task_num} = {precision * 100:.2f} %")
        with open(log_path, "a") as f:
            f.write(f"{get_timestring()},{self.train_num},{self.N},{task_success_count},{task_num},{precision}\n")

            # for i in range(len(tokens) - self.N + 1):
            #     n_gram = tuple(tokens[i:i + self.N])
            #     n_minus_1_gram = tuple(tokens[i:i + self.N - 1])
            #     self.n_grams[n_gram] += 1
            #     self.n_minus_1_grams[n_minus_1_gram] += 1
            #     if n_gram[-1] not in self.n_minus_1_grams_next_available[n_minus_1_gram]:
            #         self.n_minus_1_grams_next_available[n_minus_1_gram].append(n_gram[-1])



    # predict
    def predict_next_token(self, prev_tokens):
        """
        Predict the next token given the previous (N-1) tokens
        :param prev_tokens: A list of the previous tokens (length N-1)
        :return: The predicted next token
        """
        prev_tokens = tuple(prev_tokens[-(self.N - 1):])  # Get the most recent N-1 tokens
        # print(f"prev_tokens={prev_tokens}")
        # possible_n_grams = {n_gram: prob for n_gram, prob in self.probabilities.items() if n_gram[:-1] == prev_tokens}
        next_available_token_list = self.n_minus_1_grams_next_available[prev_tokens]
        next_rank_list = []
        if len(next_available_token_list) == 0:
            best_next_token = ";"
        else:
            for one_next_available_token in next_available_token_list:
                complete_tokens = prev_tokens + (one_next_available_token,)
                # print(f"complete_tokens={complete_tokens}")
                n_minus_1_count = self.n_minus_1_grams[prev_tokens]
                n_count = self.n_grams[complete_tokens]
                if n_minus_1_count == 0:
                    score = 0.0
                else:
                    score = n_count / n_minus_1_count
                next_rank_list.append((one_next_available_token, score))
            next_rank_list.sort(key=lambda x: -x[1])
            # print(next_rank_list)
            best_next_token = next_rank_list[0][0]
        # print(f"prev_tokens: {prev_tokens}, best_next_token = \'{best_next_token}\'")
        return best_next_token


        # if not possible_n_grams:
        #     return "</s>"  # If no match, return end token
        #
        # # Pick the next token based on probability
        # next_token = max(possible_n_grams, key=possible_n_grams.get)[-1]
        # return next_token

    # def predict_sentence(self, input_tokens):
    #     """
    #     Predict a complete sentence based on input tokens
    #     :param input_tokens: A list of starting tokens
    #     :return: The predicted sentence (as a list of tokens)
    #     """
    #     sentence = input_tokens[:]
    #     while sentence[-1] != "</s>":
    #         next_token = self.predict_next_token(sentence)
    #         sentence.append(next_token)
    #         if len(sentence) > 1200:  # Prevent infinite loop in case of bad model
    #             break
    #     return sentence[:-1]  # Remove the end token "</s>"


def run_one_task():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5, help='N of N-gram')
    args = parser.parse_args()

    train_num = 500

    n_gram_model = N_gram(args.n, train_num)
    n_gram_model.generate_datasets(
        os.path.join("data_processed", f"train_{train_num}"),
        os.path.join("data_processed", f"test"),
    )
    n_gram_model.generate_vocabulary_train()
    n_gram_model.run_test(n_gram_model.token_dataset_test)


# def run_multi_task():
#     for one_train_num in [16000]:
#         for n in range(2, 11):
#             n_gram_model = N_gram(n, one_train_num)
#             n_gram_model.generate_datasets(
#                 os.path.join("data_processed", f"train_{one_train_num}"),
#                 os.path.join("data_processed", f"test"),
#             )
#             n_gram_model.generate_vocabulary_train()
#             n_gram_model.run_test(n_gram_model.token_dataset_test)


if __name__ == "__main__":
    pass
    # run_one_task()
    # run_multi_task()
    # N = 4
    # n_gram_model = N_gram(N)
    # # next_token = n_gram_model.predict_next_token(("throws", "Exception", "{"))
    # # print(next_token)
    # n_gram_model.run_test(n_gram_model.token_dataset_test)

    # # initialize model
    # N = 5
    # model = N_gram(N)
    #
    # # generate train_token_lists from tokenized files like this:
    # # train_token_lists = [
    # #     ["the", "cat", "sat", "on", "the", "mat"],
    # #     ["the", "dog", "barked"],
    # #     ["the", "cat", "meowed"]
    # # ]
    # train_set_tokenized_folder = 'data_sets/train_set_tokenized'
    # train_token_lists = N_gram.load_tokenized_files(train_set_tokenized_folder)
    # # for per_token_list in train_token_lists:
    # #     print(per_token_list[-5:])
    # # print("length of train_token_lists is:", len(train_token_lists) )
    #
    # # train the model
    # model.generate_dict(train_token_lists)
    # # for key, value in model.n_grams.items():
    # #     print(key,"  ", value)
    # # for key, value in model.n_minus_1_grams.items():
    # #     print(key,"  ", value)
    # # for key, value in model.probabilities.items():
    # #     print(key,"  ", value)
    #
    # #test on the validation set
    # val_set_tokenized_folder = 'data_sets/val_set_tokenized'
    # val_token_lists = N_gram.load_tokenized_files(val_set_tokenized_folder)
    # # print(val_token_lists[0])
    # predict_token_lists = []
    # for test_sentence in val_token_lists:
    #     predict_sentence_list = model.predict_sentence(test_sentence[:model.N-1])
    #     print( "groundtruth: ", test_sentence )
    #     print( "input: ", test_sentence[:model.N-1] )
    #     print( "ouput: ", predict_sentence_list )
    #     predict_token_lists.append(predict_sentence_list)
    #     # print("Predicted sentence:", " ".join(predict_sentence_list))
    # print("length of val_token_lists is:", len(val_token_lists) )
    # # precision, recall, f1, accuracy = calculate_metrics(val_token_lists, predict_token_lists)


    # #test on the test set
