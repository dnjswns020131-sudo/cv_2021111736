import random
import shutil
from pathlib import Path
from PIL import Image

random.seed(42)

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
source_dir = data_dir / "PetImages"

for animal in ["Cat", "Dog"]:

    images = sorted((source_dir / animal).glob("*.jpg"))
    random.shuffle(images)

    valid_images = []

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                img.load()

            valid_images.append(image_path)

        except (OSError, ValueError):
            continue

        if len(valid_images) == 500:
            break

    if len(valid_images) < 500:
        raise RuntimeError(f"{animal} 이미지가 부족합니다.")

    # 학습 60%, 검증 20%, 테스트 20%
    datasets = {
        "train": valid_images[:300],
        "validation": valid_images[300:400],
        "test": valid_images[400:500]
    }

    for split, files in datasets.items():

        target_dir = data_dir / split / animal.lower()
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_path in files:
            shutil.copy2(image_path, target_dir / image_path.name)

        print(f"{split}/{animal}: {len(files)}장")

print("데이터 분할 완료!")