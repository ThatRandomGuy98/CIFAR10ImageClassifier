import torch
import itertools
import torch
from copy import deepcopy
from typing import Dict, Any

def train_and_validate(
    model,
    train_loader,
    val_loader,
    epochs,
    criterion,
    optimizer,
    device
    ) -> tuple[list[float], list[float], list[float]]:
    train_losses, val_losses, val_accs = [], [], []
    for epoch in range(epochs):
        model.train()  #by setting the model to training mode we enable dropout and bacth updaters
        total_train_loss = 0
        #training loop
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)  #moving the data to the GPU for better performance
            optimizer.zero_grad()  # resets the gradients from the previous step
            outputs = model(imgs)  #forward pass
            loss = criterion(outputs, labels)  #compute loss
            loss.backward()  #backspropagates the gradients
            optimizer.step() #updates the parameters
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            #validation loop
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] |" 
              f"Train Loss: {avg_train_loss:.4f} |" 
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    return train_losses, val_losses, val_accs


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

        best_epoch = max(range(len(val_accs)), key=lambda i: val_accs[i])    #fetches the accuracy of the best epoch
        final_acc = val_accs[best_epoch]    #fetches the accuracy of the best epoch
        print(f">>> Best Validation Accuracy: {final_acc:.2f}%")

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


