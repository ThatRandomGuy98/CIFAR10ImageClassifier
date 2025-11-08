import torch

def train_and_validate(model, train_loader, val_loader, epochs, criterion, optimizer, device):
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


