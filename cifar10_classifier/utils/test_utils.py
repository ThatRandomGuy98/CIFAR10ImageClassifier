import torch

def test_loop(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc