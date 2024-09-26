CSCI 680 AI for Software Engineering - N-Gram
===

---

### Authors: 
1) Enze Xu (exu03@wm.edu)
2) Yi Lin (ylin13@wm.edu).

---

# Contents

* [1 Introduction](#1-introduction)
* [2 Getting Started](#2-getting-started)
  * [2.1 Preparations](#21-preparations)
  * [2.2 Install Packages](#22-install-packages)
  * [2.3 Run N-gram](#23-run-n-gram)
* [3 Report](#3-report)
* [4 Questions](#4-questions)

---
![heatmap](https://github.com/user-attachments/assets/4e0ca210-8325-46eb-b083-9740452cd5b4)


---

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

The N-gram demo script will utilize 250 classes for the training set and 100 classes for the test set. The hyperparameter values will range from 2 to 10.

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

An example log result for the N-gram demo script is as follows:
```text
timestring,train_num,N,task_success_count,task_num,precision
20240926_010819_573420,250,2,16496,73566,0.2242340211510752
20240926_010821_567154,250,3,18965,73566,0.2577957208493054
20240926_010822_525474,250,4,16171,73566,0.21981621944920207
20240926_010823_264350,250,5,12582,73566,0.17103009542451675
20240926_010823_928997,250,6,9878,73566,0.13427398526493217
20240926_010824_587305,250,7,8370,73566,0.11377538536824076
20240926_010825_257572,250,8,7236,73566,0.09836065573770492
20240926_010825_934722,250,9,6414,73566,0.0871870157409673
20240926_010826_613846,250,10,5891,73566,0.08007775330995297
```

# 3. Report

The assignment report is available in the file report.pdf.


# 4. Questions

If you have any questions, please contact xezpku@gmail.com.


