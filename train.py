from torchvision.datasets import CIFAR10
from torchvision import transforms
from utils import train, test, train_adversarial, test_adversarial, get_saliency_metrics
from torch.utils.data import DataLoader
from model import ResNet9

train_ds = CIFAR10(
    root="/mnt/d/Calibration/",
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
        ]
    ),
    train=True,
    download=False,
)

test_ds = CIFAR10(
    root="/mnt/d/Calibration/",
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
        ]
    ),
    train=False,
    download=False,
)


model = ResNet9()
print(model.conv4[0])
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)


train_adversarial(model, train_loader, attack="FGSM", save=True)
test(model, test_loader)
test_adversarial(model, test_loader)
test_adversarial(model, test_loader, attack="FGSM")
get_saliency_metrics(model, test_ds)
