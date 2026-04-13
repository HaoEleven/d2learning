import torch
import torchvision
from torch.utils import data
from torchvision import transforms

def get_dataloader_workers():
    '''调整读取进程所需要的进程数'''
    return 4

'''获取Fashion-MNIST数据集'''
def load_data_fashion_mnist(batch_size, resize = None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root = "../data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root = "../data", train=False, transform=trans,download=True
    )

    return(
        data.DataLoader(mnist_train, batch_size, shuffle=True,
                        num_workers=get_dataloader_workers()),
        data.DataLoader(mnist_test, batch_size, shuffle=False,
                        num_workers=get_dataloader_workers())
    )