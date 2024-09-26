from ngram.N_gram import *
from ngram.utils import *


def run():
    train_num_list = [100]
    test_num = 100
    n_range = range(2, 20)

    # Prepare GitHub repos: build_dataset_files(repo_names, 0, 100) will collect about 6,857 Java classes and 312,971 Java functions.
    build_dataset_files(repo_names, 0, 2)
    # Use the javalang package to tokenize all Java files
    tokenize_data_sets(input_dir="data", output_dir="data_token")
    # Split files into the train set and test set
    split_the_files_in_data(input_dir="data_token", output_dir="data_processed", train_num_list=train_num_list, test_num=test_num)
    # Run the main experiments. The results will be saved to logs.csv.
    for one_train_num in train_num_list:
        for n in n_range:
            n_gram_model = N_gram(n, one_train_num)
            n_gram_model.generate_datasets(
                os.path.join("data_processed", f"train_{one_train_num}"),
                os.path.join("data_processed", f"test"),
            )
            n_gram_model.generate_vocabulary_train()
            n_gram_model.run_test(n_gram_model.token_dataset_test)
            del n_gram_model


if __name__ == "__main__":
    run()
