import random
import os
from collections import defaultdict
from utils import calculate_metrics

class N_gram:
    def __init__(self, N ):
        self.N = N
        self.n_grams = defaultdict(int)   # dictionary to store the N-grams tokens list and their frequencies
        self.n_minus_1_grams = defaultdict(int)    # dictionary to store the (N-1)-grams tokens list and their frequencies
        self.probabilities = {}   # dictionary to store the probabilities of each N-gram token list

    def load_tokenized_files(folder_path):
        token_lists = []
        # cnt = 0
        # Iterate over all files in the tokenized folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".java"):  # Ensure only .java files are processed
                file_path = os.path.join(folder_path, file_name)

                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code_per_file = []
                    # Read each line (token) and store it in a list for this file
                    per_file = [line.strip() for line in f.readlines() if line.strip()]
                    for line in per_file:
                        # print(each_token)
                        only_code_token = line.split(' ')[1].strip('"')
                        source_code_per_file.append(only_code_token)
                    # print(source_code_per_file[-10:])
                    token_lists.append(source_code_per_file)
                    # cnt+=1;
            # if cnt == 5:
            #     break
        return token_lists

    def generate_dict(self, token_lists):
        """
        Generate N-gram dictionaries based on tokenized input
        :param token_lists: A list of tokenized sentences (lists of tokens)
        """
        for tokens in token_lists:
            # Add start and end markers
            tokens = ['<s>'] * (self.N - 1) + tokens + ['</s>']
            for i in range(len(tokens) - self.N + 1):
                n_gram = tuple(tokens[i:i + self.N])
                n_minus_1_gram = tuple(tokens[i:i + self.N - 1])
                self.n_grams[n_gram] += 1
                self.n_minus_1_grams[n_minus_1_gram] += 1

        # Compute probabilities
        for n_gram in self.n_grams:
            n_minus_1_gram = n_gram[:-1]
            self.probabilities[n_gram] = self.n_grams[n_gram] / self.n_minus_1_grams[n_minus_1_gram]

    # predict
    def predict_next_token(self, prev_tokens):
        """
        Predict the next token given the previous (N-1) tokens
        :param prev_tokens: A list of the previous tokens (length N-1)
        :return: The predicted next token
        """
        prev_tokens = tuple(prev_tokens[-(self.N - 1):])  # Get the most recent N-1 tokens
        possible_n_grams = {n_gram: prob for n_gram, prob in self.probabilities.items() if n_gram[:-1] == prev_tokens}

        if not possible_n_grams:
            return "</s>"  # If no match, return end token

        # Pick the next token based on probability
        next_token = max(possible_n_grams, key=possible_n_grams.get)[-1]
        return next_token

    def predict_sentence(self, input_tokens):
        """
        Predict a complete sentence based on input tokens
        :param input_tokens: A list of starting tokens
        :return: The predicted sentence (as a list of tokens)
        """
        sentence = input_tokens[:]
        while sentence[-1] != "</s>":
            next_token = self.predict_next_token(sentence)
            sentence.append(next_token)
            if len(sentence) > 1200:  # Prevent infinite loop in case of bad model
                break
        return sentence[:-1]  # Remove the end token "</s>"


if __name__ == "__main__":

    # initialize model
    N = 5
    model = N_gram(N)

    # generate train_token_lists from tokenized files like this:
    # train_token_lists = [
    #     ["the", "cat", "sat", "on", "the", "mat"],
    #     ["the", "dog", "barked"],
    #     ["the", "cat", "meowed"]
    # ]
    train_set_tokenized_folder = 'data_sets/train_set_tokenized'
    train_token_lists = N_gram.load_tokenized_files(train_set_tokenized_folder)
    # for per_token_list in train_token_lists:
    #     print(per_token_list[-5:])
    # print("length of train_token_lists is:", len(train_token_lists) )

    # train the model
    model.generate_dict(train_token_lists)
    # for key, value in model.n_grams.items():
    #     print(key,"  ", value)
    # for key, value in model.n_minus_1_grams.items():
    #     print(key,"  ", value)
    # for key, value in model.probabilities.items():
    #     print(key,"  ", value)

    #test on the validation set
    val_set_tokenized_folder = 'data_sets/val_set_tokenized'
    val_token_lists = N_gram.load_tokenized_files(val_set_tokenized_folder)
    # print(val_token_lists[0])
    predict_token_lists = []
    for test_sentence in val_token_lists:
        predict_sentence_list = model.predict_sentence(test_sentence[:model.N-1])
        print( "groundtruth: ", test_sentence )
        print( "input: ", test_sentence[:model.N-1] )
        print( "ouput: ", predict_sentence_list )
        predict_token_lists.append(predict_sentence_list)
        # print("Predicted sentence:", " ".join(predict_sentence_list))
    print("length of val_token_lists is:", len(val_token_lists) )
    # precision, recall, f1, accuracy = calculate_metrics(val_token_lists, predict_token_lists)


    # #test on the test set
