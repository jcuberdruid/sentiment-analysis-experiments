import json
import os
import matplotlib.pyplot as plt
import glob

def plot_data_from_json(json_file, output_dir):
    with open(json_file, 'r') as file:
        data = json.load(file)

    epochs = range(1, len(data['epoch_history']['loss']) + 1)
    plt.figure()
    plt.plot(epochs, data['epoch_history']['loss'], label='Loss')
    plt.plot(epochs, data['epoch_history']['accuracy'], label='Accuracy')
    plt.plot(epochs, data['epoch_history']['val_accuracy'], label='Val Accuracy', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    base_filename = os.path.splitext(os.path.basename(json_file))[0]
    clean_filename = base_filename.replace("model_info_acc_", "")
    bidirectional = data['bidirectional']
    direction = "bi" if bidirectional else "mono"
    label = f"{clean_filename} ({direction})"
    
    # Remove spaces from filename
    filename = label.replace(' ', '')
    plt.savefig(os.path.join(output_dir, f'{filename}_plot.png'))
    plt.close()

    test_accuracy = data['test_accuracy']
    training_time = data['training_time']
    return epochs, data['epoch_history']['accuracy'], label, test_accuracy, training_time

def plot_all_accuracies(json_files_dir, output_dir):
    plt.figure()
    all_accuracies = []
    test_accuracy_vs_time = []

    for json_file in glob.glob(os.path.join(json_files_dir, 'model_info_acc_*.json')):
        epochs, accuracy, label, test_accuracy, training_time = plot_data_from_json(json_file, output_dir)
        all_accuracies.append((epochs, accuracy, label, test_accuracy))
        test_accuracy_vs_time.append((test_accuracy, training_time, label))

    all_accuracies.sort(key=lambda x: x[3], reverse=True)

    for epochs, accuracy, label, _ in all_accuracies:
        plt.plot(epochs, accuracy, linewidth=0.5, label=label)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.savefig(os.path.join(output_dir, 'comprehensive_accuracy_plot.png'), bbox_inches='tight')
    plt.close()

    # Scatter plot for test accuracy vs training time
    plt.figure()
    for test_acc, time, label in test_accuracy_vs_time:
        plt.scatter(time, test_acc, label=label)

    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Training Time')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.savefig(os.path.join(output_dir, 'test_accuracy_vs_time_plot.png'), bbox_inches='tight')
    plt.close()


json_files_dir = './'
output_dir = './graphs'
plot_all_accuracies(json_files_dir, output_dir)

