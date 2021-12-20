from torchvision.datasets import MNIST
from torch_geometric.datasets import MNISTSuperpixels
from datasets import get_dataset

image_dataset = MNIST('data', download=True)
dataset = MNISTSuperpixels(root='data')

dataset_list = ['MUTAG',
    'NCI1',
    'PROTEINS',
    'DD',
    'COLLAB',
    'github_stargazers',
    'IMDB-BINARY',
    'REDDIT-BINARY',
    'REDDIT-MULTI-5K']

for dataset_name in dataset_list:
    dataset = get_dataset(dataset_name, sparse=True, feat_str='deg+odeg100', root='data')
    print(dataset)
