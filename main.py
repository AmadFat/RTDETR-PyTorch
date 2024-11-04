from src.zoo.model import r18vd
import torch

if __name__ == "__main__":
    model = r18vd()
    x = torch.randn((1, 3, 640, 640))
    x = model(x, [{"labels": [], "quads": []}])
    print(model)
    print(x)