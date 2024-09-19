import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import ImageDataset, transform
from model import load_model, save_model
from torch.utils.data import DataLoader
import argparse

# 학습 함수
def train_one_epoch(model, device, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # tqdm으로 학습 진행 상황 표시
    loop = tqdm(dataloader, leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 손실 및 정확도 계산
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # tqdm에 현재 손실 및 정확도 표시
        loop.set_description(f"Loss: {running_loss / (total + 1e-6):.4f}, Accuracy: {100 * correct / total:.2f}%")
    
    return running_loss / len(dataloader), 100 * correct / total

# 평가 함수
def validate(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(dataloader), 100 * correct / total

# 학습 및 평가 로그를 플롯하는 함수
def plot_logs(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # 저장
    plt.savefig(save_path)
    plt.close()

# 메인 학습 루프
def main(args):
    # Device 설정(Your code)
    device = args.device
    
    print(f"Using device: {device}")
    
    # 데이터 로드(Your code)
    data_dir = args.data_dir
    trn_dir = os.path.join(data_dir, 'train')
    trn_ds = ImageDataset(trn_dir, transform)
    trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=32)

    val_dir = os.path.join(data_dir, 'validation')
    val_ds = ImageDataset(trn_dir, transform)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32)
    # 모델 로드(Your code)
    model = load_model(args.model_name, args.num_classes, args.pretrained)
    model.to(device)
    
    # 손실 함수 및 옵티마이저 설정(Your code)
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    # 학습 기록 저장 디렉토리 설정
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')

    # 학습 루프
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        
        # 학습(Your code)
        train_loss, train_acc = train_one_epoch(model, device, trn_dl, criterion, optimizer)

        # 검증(Your code)
        val_loss, val_acc = validate(model, device, val_dl, criterion)

        # 로그 저장
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        
        # 가장 성능이 좋은 모델 저장(Your code)
        if val_loss < best_val_loss and args.save_at_best:
            best_val_loss = val_loss
            file_dir = os.path.join(args.save_dir, 'best_model.pth')
            save_model(model, file_dir)
            
            print("Best model saved.")
    
    # # 학습 후 모델 저장
    # save_dir = os.path.join(args.save_dir, 'last_model.pth')
    # save_model(model, save_dir)
    # 학습 로그 플롯 저장
    plot_logs(train_losses, val_losses, train_accs, val_accs, os.path.join(args.save_dir, 'train_logs.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for image classification.")
    parser.add_argument('--data_dir', type=str, required=True, help="Root directory of the dataset")
    parser.add_argument('--device', type=str, default='mps', help="Device to use (e.g., 'cuda' or 'cpu')")
    parser.add_argument('--model_name', type=str, required=True, help="Model name (e.g., resnet18, alexnet, vgg16)")
    parser.add_argument('--num_classes', type=int, required=True, help="Number of classes")
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimizer (e.g., 'Adam', 'SGD')")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--save_at_best', action='store_true', help="Save the model when validation loss is the best")
    parser.add_argument('--save_dir', type=str, default='./train_logs', help="Directory to save model and logs")
    parser.add_argument('--pretrained', action='store_true', help="Use pretrained model")
    
    args = parser.parse_args()
    main(args)
