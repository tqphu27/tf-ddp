from __future__ import division, print_function

import argparse

import torch
import torch.nn.functional as F
from torch import distributed, nn
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json

from torchvision import transforms

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("{0}/{1}.log".format("logs", "pytorch-ddp")),
    logging.StreamHandler()
])

def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __str__(self):
        return '{:.6f}'.format(self.average)

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value, number):
        self.sum += value * number
        self.count += number


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

    @property
    def accuracy(self):
        return self.correct / self.count

    @torch.no_grad()
    def update(self, output, target):
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()

        self.correct += correct
        self.count += output.size(0)


class Trainer(object):

    def __init__(self, model, optimizer, train_loader, test_loader, device, log_dir):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.writer = SummaryWriter(log_dir)

    def fit(self, epochs, save_path):
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()

            print(
                'Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'test loss: {}, test acc: {}.'.format(test_loss, test_acc),
            )

            logging.info(
                'Epoch: {}/{}, train loss: {}, train acc: {}, test loss: {}, test acc: {}.'.format(
                    epoch, epochs, train_loss, train_acc, test_loss, test_acc
                )
            )

            self.writer.add_scalar('Loss/train', train_loss.average, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc.accuracy, epoch)
            self.writer.add_scalar('Loss/test', test_loss.average, epoch)
            self.writer.add_scalar('Accuracy/test', test_acc.accuracy, epoch)
        # Save the model after training
        self.save_model(save_path)
        self.writer.flush()


    def train(self):
        self.model.train()

        train_loss = Average()
        train_acc = Accuracy()

        for data, target in self.train_loader:
            data = data.to(self.device)
            target = target.to(self.device)

            output = self.model(data)
            loss = F.cross_entropy(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, target)

        return train_loss, train_acc

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        test_loss = Average()
        test_acc = Accuracy()

        for data, target in self.test_loader:
            data = data.to(self.device)
            target = target.to(self.device)

            output = self.model(data)
            loss = F.cross_entropy(output, target)

            test_loss.update(loss.item(), data.size(0))
            test_acc.update(output, target)

        return test_loss, test_acc

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model saved at {path}")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class MNISTDataLoader(data.DataLoader):

    def __init__(self, root, batch_size, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        dataset = datasets.MNIST(root, train=train, transform=transform, download=True)
        sampler = None
        if train and distributed_is_initialized():
            sampler = data.DistributedSampler(dataset)

        super(MNISTDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
        )


class COCODataset(Dataset):
    def __init__(self, images_dir, annotations_file, max_label=9, train=True):
        self.images_dir = images_dir
        self.max_label = max_label
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # Chuyển sang grayscale
            transforms.Resize((28, 28)),  # Resize về 28x28
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.train = train

        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)

        self.image_ids = [img['id'] for img in self.coco_data['images']]
        self.image_info = {img['id']: img for img in self.coco_data['images']}
        self.annotations = {img_id: [] for img_id in self.image_ids}
        for ann in self.coco_data['annotations']:
            self.annotations[ann['image_id']].append(ann)

        self.total_images = len(self.image_ids)  # Số lượng ảnh trong dataset

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_info[image_id]
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        image = Image.open(image_path)

        anns = self.annotations[image_id]

        if self.transform:
            image = self.transform(image)

        # For simplicity, assume there is only one annotation per image
        label = anns[0]['category_id']  # Use the first category_id as label

        # Filter out labels that are out of range
        if label > self.max_label:
            print(label)
            return None, None

        return image, label

    def get_all_images(self):
        all_images = []
        all_labels = []
        
        for idx in range(self.total_images):
            image, label = self.__getitem__(idx)
            if image is not None and label is not None:
                all_images.append(image)
                all_labels.append(label)
        
        if all_images:  # Check if the list is not empty
            all_images = torch.stack(all_images)
        all_labels = torch.tensor(all_labels)

        return all_images, all_labels

class COCODataloader(DataLoader):
    def __init__(self, images_dir, annotations_file, batch_size, max_label=12, train=True):
        dataset = COCODataset(images_dir, annotations_file, max_label=max_label, train=train)
        sampler = None
        if train and distributed_is_initialized():
            sampler = data.DistributedSampler(dataset)

        super(COCODataloader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
        )


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    model = Net()
    if distributed_is_initialized():
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
    else:
        model = nn.DataParallel(model)
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.root == "data":
        train_loader = MNISTDataLoader(args.root, args.batch_size, train=True)
        test_loader = MNISTDataLoader(args.root, args.batch_size, train=False)
    else:
        train_images_dir = f'{args.root}/train'
        train_annotations_file = f'{args.root}/train/_annotations.coco.json'
        valid_images_dir = f'{args.root}/valid'
        valid_annotations_file = f'{args.root}/val/_annotations.coco.json'

        if not os.path.exists(valid_images_dir):
            valid_images_dir = f'{args.root}/test'
            valid_annotations_file = f'{args.root}/test/_annotations.coco.json'

        train_loader = COCODataloader(train_images_dir, train_annotations_file, batch_size=args.batch_size, train=True)
        test_loader = COCODataloader(valid_images_dir, valid_annotations_file, batch_size=args.batch_size, train=False)
    
    trainer = Trainer(model, optimizer, train_loader, test_loader, device, log_dir=args.log_dir)
    trainer.fit(args.epochs, args.save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')
    parser.add_argument('-i',
                        '--init-method',
                        type=str,
                        default='tcp://127.0.0.1:23456',
                        help='URL specifying how to initialize the package.')
    parser.add_argument('-s', '--world-size', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('-r', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--save_path', type=str, default='models/model.pt')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = parser.parse_args()
    print(args)
    logging.info(args)
    if args.world_size > 1:
        distributed.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.rank,
        )

    run(args)


if __name__ == '__main__':
    main()
