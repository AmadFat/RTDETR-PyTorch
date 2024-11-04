from torch.utils.data import random_split
from src.data import ArmorDetection
from src.data import transforms as T


def train_armor_dataset(
        root: str,
        annFile: str,
):
    data = ArmorDetection(
        root, annFile,
        transforms = {
            T.Compose([
                T.ToTensor(),
                T.Resize(640),
            ])
        }
    )
    # train_data, val_data = random_split(data, [0.95, 0.05])
    # return train_data, val_data
    return data


def test_armor_dataset(
        root: str,
        annFile: str,
):
    test_data = ArmorDetection(
        root, annFile,
        transforms = {
            T.Compose([
                T.ToTensor(),
                T.Resize(640),
            ])
        }
    )
    return test_data
