import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)
#--------------------------------------------
import torch
from torch import nn
#--------------------------------------------
from models.cnn import CNN
from utils.data_utils import get_dataloaders, get_env
from utils.train_utils import train_and_validate, hyperparameter_search
from utils.test_utils import test_loop, predict_all, confusion_matrix_np, per_class_accuracy_from_cm
from utils.plot_utils import plot_metrics
# from utils.search_utils import hyperparameter_search
#---------------------------------------------
ROOT_DIR = get_env(var_name="DATA_ROOT")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main() -> None:
    
    print("Running on GPU" if DEVICE.type=="cuda" else "Running on CPU")
    print("--- Starting the classifier ---")
    train_loader, val_loader, test_loader = get_dataloaders(root_dir=ROOT_DIR)
    
    FAST_RUN = True
    if FAST_RUN:
        print("\nFAST RUN: True")
        param_grid = {"lr": [1e-3], "wd": [1e-4], "drop": [0.25]}
        epochs = 2

    else:
        param_grid = {
        "lr": [1e-2, 5e-3, 1e-3],
        "wd": [1e-3, 1e-4, 1e-5],
        "drop": [0.25, 0.3, 0.4]
        }
        epochs = 5
        
        
    criterion = nn.CrossEntropyLoss()
    results = hyperparameter_search(
        model_class=CNN,
        param_grid=param_grid,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=DEVICE,
        train_fn=train_and_validate,
        epochs=epochs
    )

    print("\nPlotting best model metrics...")
    train_losses, val_losses, val_accs = results["best_train_history"]
    plot_metrics(
        train_losses,
        val_losses,
        val_accs,
        save_path="experiments/figures/best_train_val.png",
        show=False
        )

    best_model = CNN(dropout=results["best_params"]["drop"]).to(DEVICE)
    best_model.load_state_dict(results["best_state"])

    print("\nEvaluating best model on test set...")
    y_true, y_pred = predict_all(model=best_model, loader=test_loader, device=DEVICE)
    test_acc = (y_true == y_pred).mean() * 100.0
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    cm = confusion_matrix_np(y_true, y_pred, n_classes=len(class_names))
    per_class = per_class_accuracy_from_cm(cm, class_names)
    per_class_sorted = sorted(per_class, key=lambda x: x[1]) 
    print("\nPer Class Accuracy:")
    for name, acc, _ in per_class_sorted:
        print(f"{name:<10} {acc*100:6.2f}%")
    
    
    

if __name__ == "__main__":
    main()
