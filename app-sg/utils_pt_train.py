import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import torchvision.transforms as tt

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score, f1_score, recall_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder


# def accuracy(labels, outputs):
#     _, preds = torch.max(outputs, dim=1)
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def acc_sc(labels, preds):
    return accuracy_score(labels, preds)


def f1_sc(labels, preds):
    return f1_score(labels, preds, average='weighted')


def recall_sc(labels, preds):
    return recall_score(labels, preds, average='weighted')


def auc_roc_sc(labels, preds):
    enc = OneHotEncoder(sparse=False)
    enc.fit([[0], [1], [2], [3], [4], [5], [6]])
    one_hot_true = enc.transform(np.array(labels).reshape(-1, 1))
    one_hot_pred = enc.transform(np.array(preds).reshape(-1, 1))
    return roc_auc_score(one_hot_true, one_hot_pred, average='weighted')


def precision_sc(labels, preds):
    return precision_score(labels, preds, average='weighted')


# model utils

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        _, preds = torch.max(out, dim=1)
        return [loss, preds, labels]

    # def validation_step(self, batch):
    #     images, labels = batch
    #     out = self(images)
    #     loss = F.cross_entropy(out, labels)
    #     acc = accuracy(labels, out)
    #     return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.3f}, train_acc: {:.3f}".format(
            epoch, result['train_loss'], result['train_acc']))


class ResNet(ImageClassificationBase):  # ResNet9
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.input = conv_block(in_channels, 64)

        self.conv1 = conv_block(64, 64, pool=True)
        self.res1 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = conv_block(64, 64, pool=True)
        self.res2 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = conv_block(64, 64, pool=True)
        self.res3 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop3 = nn.Dropout(0.5)

        self.classifier = nn.Sequential(nn.MaxPool2d(6),
                                        nn.Flatten(),
                                        nn.Linear(64, num_classes))

    def forward(self, xb):
        out = self.input(xb)

        out = self.conv1(out)
        out = self.res1(out) + out
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.drop3(out)

        return self.classifier(out)


class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNetBase(ImageClassificationBase):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNetBase, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, intermediate_channels * self.expansion, kernel_size=1, stride=stride),
            nn.BatchNorm2d(intermediate_channels * self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion  # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels))  # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)


def ResNet18_pt(img_channels=3, num_classes=1000):
    return ResNetBase(18, Block, img_channels, num_classes)


def ResNet34_pt(img_channels=3, num_classes=1000):
    return ResNetBase(34, Block, img_channels, num_classes)


def ResNet50_pt(img_channels=3, num_classes=1000):
    return ResNetBase(50, Block, img_channels, num_classes)


def ResNet101_pt(img_channels=3, num_classes=1000):
    return ResNetBase(101, Block, img_channels, num_classes)


def ResNet152_pt(img_channels=3, num_classes=1000):
    return ResNetBase(152, Block, img_channels, num_classes)


# training utils

# @torch.no_grad()
# def evaluate(model, train_loader):
#     model.eval()
#     batch = next(iter(train_loader))
#     outputs = model.validation_step(batch)
#     return model.validation_epoch_end(outputs)


# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


# def make_data_loaders_pt(data_dir):
#     device = get_default_device()
#     # torch.cuda.empty_cache()
#     train_tfms = tt.Compose([tt.Resize((64, 64)),
#                              tt.Grayscale(num_output_channels=1),
#                              tt.RandomHorizontalFlip(),
#                              tt.RandomRotation(30),
#                              tt.ToTensor()])
#     valid_tfms = tt.Compose([tt.Resize((64, 64)),
#                              tt.Grayscale(num_output_channels=1),
#                              tt.ToTensor()])
#     train_ds = ImageFolder(data_dir + '/test', train_tfms)
#     valid_ds = ImageFolder(data_dir + '/dev', valid_tfms)
#     batch_size = 64
#     train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
#     valid_dl = DataLoader(valid_ds, batch_size * 2, num_workers=2, pin_memory=True)
#     train_dl = DeviceDataLoader(train_dl, device)
#     valid_dl = DeviceDataLoader(valid_dl, device)
#     return train_dl, valid_dl, device


def make_train_loader_pt(data_dir, batch_size):
    device = get_default_device()
    try:
        torch.cuda.empty_cache()
    except:
        pass
    train_tfms = tt.Compose([tt.Resize((64, 64)),
                             tt.Grayscale(num_output_channels=1),
                             tt.RandomHorizontalFlip(),
                             tt.RandomRotation(30),
                             tt.ToTensor()])
    train_ds = ImageFolder(data_dir, train_tfms)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    train_dl = DeviceDataLoader(train_dl, device)
    return train_dl, device

# def train_cycle_pt(epochs, max_lr, model, train_loader, val_loader, window,
#                    weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
#     torch.cuda.empty_cache()
#     history = []
#
#     # Set up custom optimizer with weight decay
#     optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
#     # Set up one-cycle learning rate scheduler
#     sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
#                                                 steps_per_epoch=len(train_loader))
#
#     for epoch in range(epochs):
#         # Training Phase
#         model.train()
#         train_losses = []
#         predss = []
#         labelss = []
#
#         lrs = []
#         for batch in train_loader:
#             data = model.training_step(batch)
#             loss = data[0]
#             preds = data[1].cpu()
#             labels = data[2].cpu()
#             predss.extend(preds)
#             labelss.extend(labels)
#             train_losses.append(loss)
#             loss.backward()
#
#             # Gradient clipping
#             if grad_clip:
#                 nn.utils.clip_grad_value_(model.parameters(), grad_clip)
#
#             optimizer.step()
#             optimizer.zero_grad()
#
#             # Record & update learning rate
#             lrs.append(get_lr(optimizer))
#             sched.step()
#
#         # Validation phase
#         result = evaluate(model, val_loader)
#         result['train_loss'] = torch.stack(train_losses).mean().item()
#         result['train_acc'] = acc_sc(labelss, predss).item()
#         result['train_f1'] = f1_sc(labelss, predss).item()
#         result['train_racall'] = recall_sc(labelss, predss).item()
#         result['train_auc_roc'] = auc_roc_sc(labelss, predss).item()
#         result['train_precision'] = precision_sc(labelss, predss).item()
#         result['lrs'] = lrs
#         model.epoch_end(epoch, result)
#         history.append(result)
#     return history


# if __name__ == "__main__":
#     # model = ResNet18(img_channels=1, num_classes=7)
#     # image_path = "example_images/sad1.png"
#     # img = cv2.imread(image_path)
#     # img = np.asarray(img)
#     #
#     # preprocess = tt.Compose([tt.Resize((64, 64)),
#     #                          tt.Grayscale(num_output_channels=1),
#     #                          tt.ToTensor()])
#     #
#     # img_preprocessed = preprocess(Image.fromarray(img))
#     # batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)
#     # out = model(batch_img_tensor).to("cuda")
#     # percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100  # procenty
#     # print(percentage.tolist())
#     print("Detected device:")
#     print(get_default_device())
#
#     data_dir = 'C:/Users/Aleksander Podsiad/Desktop/Projekt Emocje/Dane'
#     # lokalizacja folderu z danymi test, train i dev, jako train używam test bo inaczej zamula
#     train_dl, valid_dl, device = make_data_loaders_pt(data_dir)
#     history = []
#     model = to_device(ResNet18(1, 7), device)
#     epochs = 3
#     max_lr = 0.001
#     grad_clip = 0.2
#     weight_decay = 1e-4
#     opt_func = torch.optim.SGD  # torch.optim.Adam
#     history += train_cycle_pt(epochs, max_lr, model, train_dl, valid_dl,
#                               grad_clip=grad_clip, weight_decay=weight_decay, opt_func=opt_func)
#     train_accs = [x['train_acc'] for x in history]
#     print(train_accs)
