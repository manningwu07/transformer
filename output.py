import pandas as pd
import matplotlib.pyplot as plt

def plot_training_metrics(csv_file):
    """
    Reads a CSV file containing training metrics and plots them.
    The CSV file is expected to have 'epoch', 'accuracy', and 'loss' columns.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return

    # Check if required columns exist
    if 'epoch' not in df.columns or 'accuracy' not in df.columns or 'loss' not in df.columns:
        print("Error: The CSV file must contain 'epoch', 'accuracy', and 'loss' columns.")
        return

    epochs = df['epoch']
    accuracy = df['accuracy']
    loss = df['loss']

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot Accuracy
    ax1.plot(epochs, accuracy, marker='o', linestyle='-', color='b')
    ax1.set_title('Training Accuracy per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    ax1.set_xticks(epochs)

    # Plot Loss
    ax2.plot(epochs, loss, marker='o', linestyle='-', color='r')
    ax2.set_title('Training Loss per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.set_xticks(epochs)

    plt.tight_layout() # Adjusts subplot params for a tight layout
    plt.show()

if __name__ == '__main__':
    plot_training_metrics('training_log.csv')