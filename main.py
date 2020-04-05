# -----------------------------------------------------------
# main.py 2020-03-08
#
# Define entry point for program
#
# Copyright (c) 2020-2021 Team Artificial Incompetence, COMP 472
# All rights reserved.
# -----------------------------------------------------------

from byom.lidstone import lidstone
from byom.majority_vote import majority_vote_cl
from constants import *
from performance.model_selection import evaluate_hyperparameters
from required_model import required_model


def main():
    """
    Entry point of program.
    """
    print("Welcome to the Artificial Incompetence Team's Project 2 submission.")
    print("You can:",
          "\n(1) Run the required model (not BYOM)",
          "\n(2) Run the required model against a grid of hyper-parameters to find the best combination",
          "\n(3) Run the Lidstone BYOM",
          "\n(4) Run the Majority vote BYOM")
    choice = int(input("Your choice: "))

    v = 0
    n = 1
    delta = 0
    if choice in [1, 3, 4]:
        v = int(input("Vocabulary (0, 1 or 2) (default: 0): ") or 0)
        n = int(input("Ngram size (1, 2 or 3) (default: 1): ") or 1)
        delta = float(input("Delta (0.0 to 1.0) (default: 0): ") or 0)

    train_file = input("Location of training tweets. (default: '{}'): ".format(
        DEFAULT_TRAINING_FILE)) or DEFAULT_TRAINING_FILE
    test_file = input("Location of training tweets. (default: '{}'): ".format(
        DEFAULT_TEST_FILE)) or DEFAULT_TEST_FILE

    if choice == 1:
        required_model(v, n, delta, train_file, test_file)
    if choice == 2:
        evaluate_hyperparameters(train_file, test_file)
    if choice == 3:
        lidstone(v, n, delta, train_file, test_file)
    if choice == 4:
        majority_vote_cl(v, n, delta, train_file, test_file)


main()
