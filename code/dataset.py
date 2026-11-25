from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

train_dataset = datasets.ImageFolder(
    data_dir / "train", transform=transform
)

val_dataset = datasets.ImageFolder(
    data_dir / "validation", transform=transform
)

test_dataset = datasets.ImageFolder(
    data_dir / "test", transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True
)

val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False
)

test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False
)

# 데이터 확인
if __name__ == "__main__":
    print("학습 데이터:", len(train_dataset))
    print("검증 데이터:", len(val_dataset))
    print("테스트 데이터:", len(test_dataset))
    print("클래스:", train_dataset.class_to_idx)

    images, labels = next(iter(train_loader))

    print("이미지 크기:", images.shape)
    print("라벨 크기:", labels.shape)