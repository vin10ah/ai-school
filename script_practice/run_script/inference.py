import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import load_model
import argparse

# 예측 함수
def predict(model, device, image_path, transform):
    # 이미지 로드 및 전처리
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # 배치 차원을 추가
    image_tensor = image_tensor.to(device)
    
    # 예측 수행
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item(), image

# 이미지 및 예측된 클래스 반환 함수
def display_prediction(image, predicted_class_name):
    plt.imshow(image)
    plt.title(f"Predicted Class: {predicted_class_name}")
    plt.axis('off')  # 축을 없애고 이미지에만 집중
    plt.show()

def main(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Transform 정의 (모델 학습 시 사용한 것과 동일한 변환 적용)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 모델 로드
    model = load_model(args.model_name, args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path))  # 학습된 모델 가중치 로드
    model = model.to(device)
    
    # 입력된 이미지에 대한 예측 수행
    predicted_class_index, image = predict(model, device, args.image_path, transform)
    
    # 이미지와 예측된 클래스 표시
    display_prediction(image, predicted_class_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on a single image and display the result.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the image to perform inference on")
    parser.add_argument('--data_dir', type=str, required=True, help="Root directory of the dataset (to retrieve class names)")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (e.g., 'cuda', 'cpu', 'mps')")
    parser.add_argument('--model_name', type=str, required=True, help="Model name (e.g., resnet18, alexnet, vgg16)")
    parser.add_argument('--num_classes', type=int, required=True, help="Number of classes")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file (e.g., 'best_model.pth')")
    
    args = parser.parse_args()
    main(args)
