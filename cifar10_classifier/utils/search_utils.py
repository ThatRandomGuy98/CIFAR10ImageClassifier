import itertools
import torch
from copy import deepcopy
from typing import Dict, Any

def hyperparameter_search(
    model_class,
    param_grid,
    train_loader,
    val_loader,
    criterion,
    device,
    train_fn,
    epochs=5,
    optimizer_class=torch.optim.Adam
) -> Dict[str, Any]:
    """
    Performs grid search over the given hyperparameter combinations.
    Returns the best model state, params, validation accuracy, and metrics history.
    """
    best_val_acc = 0
    best_params = None
    best_model_state = None
    best_train_history = None  
    # --- Iterate over all possible hyperparameter combinations ---
    for lr, wd, drop in itertools.product(param_grid["lr"], param_grid["wd"], param_grid["drop"]):
        print(f"\n--- Testing configuration: lr={lr}, wd={wd}, dropout={drop} ---")
        model = model_class(dropout=drop).to(device)
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=wd)

        train_losses, val_losses, val_accs = train_fn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        final_acc = val_accs[-1]    #fetches the accuracy of the final epoch
        print(f">>> Final Validation Accuracy: {final_acc:.2f}%")

        # Store the best configuration
        if final_acc > best_val_acc:
            best_val_acc = final_acc
            best_params = {"lr": lr, "wd": wd, "drop": drop}
            best_model_state = deepcopy(model.state_dict()) # deep copy to avoid reference overwrite
            best_train_history = (train_losses, val_losses, val_accs)

    print("\n>>> Best hyperparameters found:")
    print(f"   Learning rate: {best_params['lr']}")
    print(f"   Weight decay:  {best_params['wd']}")
    print(f"   Dropout:       {best_params['drop']}")
    print(f"   Validation accuracy: {best_val_acc:.2f}%")

    return {
        "best_state": best_model_state,
        "best_params": best_params,
        "best_val_acc": best_val_acc,
        "best_train_history": best_train_history,
    }