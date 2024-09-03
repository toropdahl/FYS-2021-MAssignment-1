import pandas as pd
import numpy as np
from problem_1 import csv_to_test_and_train
import math 
X_train, X_test, y_train, y_test = csv_to_test_and_train()


print(f"Training set size: {len(y_train)} samples, Test set size: {len(y_test)} samples")
print(f"Training set distribution: Pop = {np.sum(y_train)}, Classical = {len(y_train) - np.sum(y_train)}")
print(f"Test set distribution: Pop = {np.sum(y_test)}, Classical = {len(y_test) - np.sum(y_test)}")



w1 = np.random.uniform(-0.01, 0.01)
w2 = np.random.uniform(-0.01, 0.01)

b = 0

learning_rate = 0.0022

def sigmoid(x1, x2):
    z = (w1 * x1) + (w2 * x2) + b
    y = 1/(1+(math.exp(-z)))
    return y

def cost_single_sample(yi, y_hat):
    if yi == 1:
        return -math.log(y_hat)  
    else:
        return -math.log(1 - y_hat)


def sdg_step(x1, x2, yi, y_hat):
    global w1, w2, b, learning_rate
    error = y_hat - yi
    w1_grad = error * x1
    w2_grad = error * x2
    b_grad = error
    w1 -= learning_rate * w1_grad
    w2 -= learning_rate * w2_grad
    b = b - learning_rate * b_grad

def epoch_train():
    index = 0
    total_cost = 0
    for x1,x2 in X_train:
        yi = y_train[index]
        index+=1
        y_hat = sigmoid(x1, x2)
        cost = cost_single_sample(yi, y_hat)
        total_cost+=cost
        sdg_step(x1,x2,yi,y_hat)
        
    """     if (index-1)%100 == 0:
            print(f"Cost on sample {index-1}: {cost}")
     """
    print(f"The total cost of epoch is : {total_cost}")

def shuffle_n_train(epochs):
    global X_train, y_train  # Use global to ensure the function accesses the global variables

    for epoch in range(epochs):
        # Step 1: Shuffle the data
        indices = np.random.permutation(len(X_train))  # Get a shuffled sequence of indices
        X_train = X_train[indices]  # Shuffle X_train using the shuffled indices
        y_train = y_train[indices]  # Shuffle y_train using the shuffled indices
        
        # Step 2: Train for one epoch
        print(f"\nEpoch {epoch + 1}")
        epoch_train()


def test_loop():
    total_cost = 0
    correct_predictions = 0
    total_samples = len(X_test)

    for i in range(total_samples):
        x1, x2 = X_test[i]
        yi = y_test[i]


        y_hat = sigmoid(x1, x2)


        cost = cost_single_sample(yi, y_hat)
        total_cost += cost


        predicted_label = 1 if y_hat >= 0.5 else 0

        if predicted_label == yi:
            correct_predictions += 1

    # Calculate the average cost for the test set
    average_cost = total_cost / total_samples

    # Calculate the accuracy as the percentage of correct predictions
    accuracy = (correct_predictions / total_samples) * 100

    # Print the results
    print(f"Total cost on test set: {total_cost}")
    print(f"Average cost on test set: {average_cost}")
    print(f"Accuracy on test set: {accuracy:.2f}%")



if __name__ == "__main__":
    shuffle_n_train(20)
    test_loop()

