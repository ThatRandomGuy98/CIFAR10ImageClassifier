import torch
from torch import nn
#--------------------------------------------
from models.cnn import CNN
from utils.data_utils import get_dataloaders, get_env
from utils.train_utils import train_and_validate
from utils.test_utils import test_loop
from utils.plot_utils import plot_metrics
from utils.search_utils import hyperparameter_search
#---------------------------------------------
# import os
# from dotenv import load_dotenv
# load_dotenv()
# DATA_ROOT = os.getenv("DATA_ROOT")
# print("DATA_ROOT:", os.getenv("DATA_ROOT"))
# if not DATA_ROOT:
#     raise EnvironmentError(
#         "Could not find DATA_ROOT, make sure the .env file is ok"
#     )
ROOT_DIR = get_env(name="DATA_ROOT")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    
    print("--- Starting the classifier ---")
    train_loader, val_loader, test_loader = get_dataloaders(root_dir=ROOT_DIR)
    param_grid = {
        "lr": [1e-2, 5e-3, 1e-3],
        "wd": [1e-3, 1e-4, 1e-5],
        "drop": [0.25, 0.3, 0.4]
    }
    criterion = nn.CrossEntropyLoss()
    results = hyperparameter_search(
        model_class=CNN,
        param_grid=param_grid,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=DEVICE,
        train_fn=train_and_validate,
        epochs=5
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
    test_acc = test_loop(best_model, test_loader, device=DEVICE)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
