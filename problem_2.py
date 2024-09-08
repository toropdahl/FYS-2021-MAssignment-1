import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from problem_1 import csv_to_test_and_train
from config import learning_rate, epoch_number, want_plot

# Load training and test data
X_train, X_test, y_train, y_test = csv_to_test_and_train()

# Initialize weights, bias, and training error list
w1 = np.random.uniform(-0.01, 0.01)
w2 = np.random.uniform(-0.01, 0.01)
b = 0

# List to store error values for each epoch
epoch_errors = []

def sigmoid(x1, x2):
    z = (w1 * x1) + (w2 * x2) + b
    return 1 / (1 + math.exp(-z))

def cost_single_sample(yi, y_hat):
    if yi == 1:
        return -math.log(y_hat)
    else:
        return -math.log(1 - y_hat)

def sdg_step(x1, x2, yi, y_hat):
    global w1, w2, b, learning_rate
    error = y_hat - yi
    w1 -= learning_rate * error * x1
    w2 -= learning_rate * error * x2
    b -= learning_rate * np.mean(error)

def epoch_train():
    total_cost = 0
    for index in range(len(X_train)):
        x1, x2 = X_train[index]
        yi = y_train[index]
        y_hat = sigmoid(x1, x2)
        cost = cost_single_sample(yi, y_hat)
        total_cost += cost
        sdg_step(x1, x2, yi, y_hat)
    
    # Calculate and store the average cost for this epoch
    average_cost = total_cost / len(X_train)
    epoch_errors.append(average_cost)
    print(f"The average cost after this epoch is: {average_cost}")

def shuffle_n_train(epochs):
    global X_train, y_train

    for epoch in range(epochs):
        # Shuffle the data
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        # Train for one epoch
        print(f"\nEpoch {epoch + 1}")
        epoch_train()

def test_loop():
    total_cost = 0
    total_samples = len(X_test)

    # Initialize confusion matrix counters
    TP = 0  # True Positives
    TN = 0  # True Negatives
    FP = 0  # False Positives
    FN = 0  # False Negatives

    for i in range(total_samples):
        x1, x2 = X_test[i]
        yi = y_test[i]

        # Compute predicted probability
        y_hat = sigmoid(x1, x2)

        # Calculate the cost for the current sample
        cost = cost_single_sample(yi, y_hat)
        total_cost += cost

        # Determine the predicted label
        predicted_label = 1 if y_hat >= 0.5 else 0

        # Update confusion matrix counters
        if predicted_label == 1 and yi == 1:
            TP += 1
        elif predicted_label == 0 and yi == 0:
            TN += 1
        elif predicted_label == 1 and yi == 0:
            FP += 1
        elif predicted_label == 0 and yi == 1:
            FN += 1

    # Calculate the average cost for the test set
    average_cost = total_cost / total_samples

    # Calculate the accuracy as the percentage of correct predictions
    accuracy = (TP + TN) / total_samples * 100

    # Print the results
    print(f"Total cost on test set: {total_cost}")
    print(f"Average cost on test set: {average_cost}")
    print(f"Accuracy on test set: {accuracy:.2f}%")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f"                Predicted Positive (1)    Predicted Negative (0)")
    print(f"Actual Positive (1)     {TP}                       {FN}")
    print(f"Actual Negative (0)     {FP}                       {TN}")


def plot_errors(run):
    if run:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch_number + 1), epoch_errors, label='Training Error', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Average Training Error')
        plt.title('Training Error Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    shuffle_n_train(epoch_number)
    test_loop()
    plot_errors(want_plot)

    
