CSCI 680 AI for Software Engineering - N-Gram
===


# Contents

* [1 Introduction](#1-introduction)
* [2 Getting Started](#2-getting-started)
  * [2.1 Preparations](#21-preparations)
  * [2.2 Install Packages](#22-install-packages)
  * [2.3 Run N-gram](#23-run-n-gram)
* [3 Questions](#3-questions)



# 1. Introduction
Code completion in Java aims to automatically complete the code for Java methods or classes. The N-gram is a language model that can predict the next token in a sequence by learning the probabilities of token sequences based on their occurrences in the training data and choosing the token with the highest probability to follow.


# 2. Getting Started

This project is developed using Python 3.9+ and is compatible with macOS or Linux

## 2.1 Preparations

(1) Clone the repository to your workspace.

```shell
~ $ git clone https://github.com/EnzeXu/CSCI680_N_Gram.git
```

(2) Navigate into the repository.
```shell
~ $ cd CSCI680_N_Gram
~/CSCI680_N_Gram $
```

(3) Create a new virtual environment and activate it. In this case we use Virtualenv environment (Here we assume you have installed the `virtualenv` package using you source python script), you can use other virtual environments instead (like conda).

For macOS or Linux operating systems:
```shell
~/CSCI680_N_Gram $ python -m venv ./venv/
~/CSCI680_N_Gram $ source venv/bin/activate
(venv) ~/CSCI680_N_Gram $ 
```

For Windows operating systems:

```shell
~/CSCI680_N_Gram $ python -m venv ./venv/
~/CSCI680_N_Gram $ .\venv\Scripts\activate
(venv) ~/CSCI680_N_Gram $ 
```

You can use the command deactivate to exit the virtual environment at any time.

## 2.2 Install Packages

```shell
(venv) ~/CSCI680_N_Gram $ pip install -r requirements.txt
```

## 2.3 Run N-gram

(1) Run N-gram demo

The N-gram demo script will utilize 100 classes for the training set and 100 classes for the test set. The hyperparameter values will range from 2 to 10.

```shell
(venv) ~/CSCI680_N_Gram $ python run_demo.py
```

(2) Run N-gram complete

The N-gram complete script will utilize 125 / 250 / 500 / 1,000 / 2,000 / 4,000 / 8,000 / 16,000 classes for the training set and 100 classes for the test set. The hyperparameter values will range from 2 to 10. This experiment setting is the same as the results introduced in the final report.

```shell
(venv) ~/CSCI680_N_Gram $ python run_complete.py
```



(3) Collect the results from `logs.txt`.

```shell
(venv) ~/CSCI680_N_Gram $ cat logs.txt
```



# 3. Questions

If you have any questions, please contact xezpku@gmail.com.


