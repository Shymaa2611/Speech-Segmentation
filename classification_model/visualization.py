import matplotlib.pyplot as plt

def plot_loss(train_losses,val_losses):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='teal')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


def plot_accuracy(train_acc,val_acc):
   plt.subplot(1, 2, 2)
   plt.plot(train_acc, label='Train Accuracy', color='teal')
   plt.plot(val_acc, label='Validation Accuracy', color='orange')
   plt.title('Training and Validation Accuracy')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.tight_layout()
   plt.show()