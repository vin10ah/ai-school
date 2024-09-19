import torch
import torchvision.models as models
import argparse

def load_model(model_name, num_classes, pretrained=False):
    """
    모델을 불러오는 함수
    :param model_name: 사용할 모델 이름 (resnet18, alexnet 등)
    :param num_classes: 출력 클래스 수
    :param pretrained: 사전 학습된 가중치를 사용할지 여부
    :return: 정의된 모델
    """
    if model_name == 'resnet18':
        model = models.resnet18(weights=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'alexnet':
        model = models.alexnet(weights=pretrained)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=pretrained)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model

def save_model(model, path):
    """모델을 저장하는 함수"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_saved_model(model_name, num_classes, path):
    """저장된 모델을 불러오는 함수"""
    model = load_model(model_name, num_classes)
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

def main(args):
    # 모델을 불러오기
    model = load_model(args.model_name, args.num_classes, weights=args.pretrained)
    
    # 모델을 저장하거나 저장된 모델 불러오기(Your code)
    if args.save_model:
        save_model(model, args.save_model)
    else:
        load_saved_model(args.model, args.num_classes, args.load_model)
    
    print(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load or save a model")
    parser.add_argument('--model_name', type=str, required=True, help='Model name: resnet18, alexnet, vgg16')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--save_model', type=str, help='Path to save the model')
    parser.add_argument('--load_model', type=str, help='Path to load a saved model')
    
    args = parser.parse_args()
    main(args)
