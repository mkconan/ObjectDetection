from torchvision.datasets import CocoDetection
from torchvision import transforms

coco_dataset = CocoDetection(
    root="./data/coco/images",
    annFile="./data/coco/annotations/instances_train2017.json",
    transform=transforms.ToTensor(),
)
