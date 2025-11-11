import matplotlib.pyplot as plt
import os

def plot_metrics(train_losses, val_losses, val_accs, show=False, save_path=None) -> None:
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, color='green', label='Val Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"--- Saved plot to {save_path} ---")
        
    if show:
        plt.show()
    else:
        plt.close()
