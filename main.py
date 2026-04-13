from load_data import load_data_fashion_mnist
from train import train
from Net.LeNet import net
from d2l import torch as d2l

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size = batch_size)

    lr, num_epochs = 0.9, 10
    train(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
