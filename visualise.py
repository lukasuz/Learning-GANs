import matplotlib.pyplot as plt
import numpy as np
import re
from math import ceil

mode_types = ["cnn", "dense", "conditional"]
path = "./logs/"

def extract_data(path):
    with open(path, 'r') as f:
        data = f.read().strip()
    data = re.split(r',|\n', data)

    tag1 = data[0]
    tag2 = data[1]

    data1 = np.array(data[2::2], dtype='float32')
    data2 = np.array(data[3::2], dtype='float32')
    
    return data1, data2

def visualise_mode_type(mode_type):

    d_path = "{0}{1}_discriminator.txt".format(path, mode_type)
    g_path = "{0}{1}_gan.txt".format(path, mode_type)

    try:
        d_loss, d_acc = extract_data(d_path)
        g_loss, g_acc = extract_data(g_path)
        print("Showing results for \"{0}\"".format(mode_type))
    except FileNotFoundError as e:
        print("Not data for \"{0}\" available yet.".format(mode_type))
        return
    
    x = np.arange(len(d_loss))

    fig, axs = plt.subplots(2)
    fig.suptitle(mode_type)

    axs[0].plot(d_loss, label="Discriminator Loss")
    axs[0].plot(g_loss, label="Gan Loss")
    axs[0].set_title("Loss")
    axs[0].legend(["Discriminator", "Gan"])

    axs[1].plot(x, d_acc)
    axs[1].plot(x, g_acc)
    axs[1].set_title("Accuracy")
    axs[1].legend(["Discriminator", "Gan"])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for type in  mode_types:
        visualise_mode_type(type)
    
    