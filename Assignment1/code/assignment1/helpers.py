import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def to_one_hot(labels: npt.NDArray[int], n_classes: int) -> npt.NDArray[int]:
    return np.eye(n_classes)[labels]


def save_loss_curve(train_losses, val_accuracies, epochs, filename = 'loss_curve_numpy.png'):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(epochs), train_losses, color='tab:red', marker='o', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='tab:blue')
    ax2.plot(
            range(epochs),
            val_accuracies,
            color='tab:blue',
            marker='s',
            label='Validation Accuracy'
    )
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Loss and Validation Accuracy')
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))

    plt.savefig(filename)
