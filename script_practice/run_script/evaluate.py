import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import ImageDataset, transform
from model import load_saved_model
import argparse
from tqdm import tqdm
from glob import glob

# 평가 함수
def evaluate(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return test_loss, accuracy

def main(args):
    # Device 설정(Your code)
    device = args.device
    
    print(f"Using device: {device}")
    
    # 테스트 데이터 로드(Your code)
    tst_dir = os.path.join(args.data_dir, 'test')
    tst_ds = ImageDataset(tst_dir, transform)
    tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size=32)
    
    # 모델 로드(Your code)
    model = load_saved_model(args.model_name, args.num_classes, args.model_path)
    model.to(device)
    
    # 손실 함수 정의
    criterion = nn.CrossEntropyLoss()
    
    # 평가
    test_loss, accuracy = evaluate(model, device, tst_dl, criterion)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test dataset.")
    parser.add_argument('--data_dir', type=str, required=True, help="Root directory of the dataset")
    parser.add_argument('--device', type=str, default='mps', help="Device to use (e.g., 'cuda' or 'cpu')")
    parser.add_argument('--model_name', type=str, required=True, help="Model name (e.g., resnet18, alexnet, vgg16)")
    parser.add_argument('--num_classes', type=int, required=True, help="Number of classes")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file (e.g., 'best_model.pth')")
    
    args = parser.parse_args()
    main(args)
