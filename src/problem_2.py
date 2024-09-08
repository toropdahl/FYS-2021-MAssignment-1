import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
from src.problem_1 import csv_to_test_and_train
from src.config import learning_rate, epoch_number, want_plot

# Load training and test data
X_train, X_test, y_train, y_test = csv_to_test_and_train()

# Initialize weights and bias
w1 = np.random.uniform(-0.01, 0.01)
w2 = np.random.uniform(-0.01, 0.01)
b = 0

# List to store error values for each epoch
epoch_errors = []

def sigmoid(x1, x2):
    """
    Function for calculating a probability for a single sample
    """
    z = (w1 * x1) + (w2 * x2) + b
    return 1 / (1 + math.exp(-z))

def cost_single_sample(yi, y_hat):
    """
    Function for calculating the cost of a single sample
    """
    if yi == 1:
        return -math.log(y_hat)
    else:
        return -math.log(1 - y_hat)

def sdg_step(x1, x2, yi, y_hat):
    """
    Function for updating weights and bias after calculating probability of a single sample
    """
    global w1, w2, b, learning_rate
    error = y_hat - yi
    w1 -= learning_rate * error * x1
    w2 -= learning_rate * error * x2
    b -= learning_rate * error

def epoch_train():
    """
    Uses the global X_train and y_train to run though 1 epoch of training
    """
    total_cost = 0
    for index in range(len(X_train)):
        x1, x2 = X_train[index]
        yi = y_train[index]
        y_hat = sigmoid(x1, x2)
        sdg_step(x1, x2, yi, y_hat)
        cost = cost_single_sample(yi, y_hat)
        total_cost += cost
    
    # Calculate and store the average cost for this epoch
    average_cost = total_cost / len(X_train)
    epoch_errors.append(average_cost)
    # print(f"The average cost after this epoch is: {average_cost}")

def shuffle_n_train(epochs):
    """
    Shuffles the global X_train and y_train, and uses the epoch_train function to train the model on the selected number of epochs.
    """
    global X_train, y_train

    for epoch in range(epochs):
        # Shuffle the data
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        # Train for one epoch
        # print(f"\nEpoch {epoch + 1}")
        epoch_train()

def test_loop():
    """
    Function for testing the model. Uses training data set and displays the result
    """
    total_cost = 0
    total_samples = len(X_test)

    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

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

        # Store true and predicted labels for confusion matrix
        true_labels.append(yi)
        predicted_labels.append(predicted_label)

    # Calculate the average cost for the test set
    average_cost = total_cost / total_samples

    # Calculate the accuracy as the percentage of correct predictions
    accuracy = (np.array(true_labels) == np.array(predicted_labels)).mean() * 100

    # Print the results
    print("\nTest prints:")
    print(f"Total cost on test set: {total_cost}")
    print(f"Average cost on test set: {average_cost}")
    print(f"Accuracy on test set: {accuracy:.2f}%")

    # Display confusion matrix using the dedicated function
    
    cm = confusion_matrix(true_labels, predicted_labels)
    # Print confusion matrix with improved formatting
    print("\nConfusion Matrix:")
    print(cm)

    
    

def plot_errors(run):
    """
    Function used to plot the training errors over the epochs. uses variable which is changed in config to choose if the plot should run or not.
    """
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

    
