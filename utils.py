import torch as ch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os


class VggFeatureExtractor(nn.Module):
    def __init__(self, base_model, num_features):
        super(VggFeatureExtractor, self).__init__()
        self.featurization_model = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_features),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(num_features, 1),
        )

    def forward(self, x, with_latent=False):
        # Extract features without back-prop
        with ch.no_grad():
            x = self.featurization_model(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if with_latent and i == 5:
                latent = x
        if with_latent:
            return (latent, x)
        return x


class MultipleModelsWrapper:
  def __init__(self, models, clf, latent, n_gpus=4):
    self.models = models
    self.clf = clf
    self.latent= latent
    self.n_gpus = n_gpus

  def __call__(self, x):
    features = []
    x_ = [x.to('cuda:%d' % i) for i in range(self.n_gpus)]
    for i, model in enumerate(self.models):
      with ch.no_grad():
        if self.latent:
          score = model(x_[i % self.n_gpus], with_latent=True)[0].cpu()
        else:
          score = model(x_[i % self.n_gpus]).cpu()
      features.append(score)
    features = ch.stack(features, 0).numpy()
    features = features.transpose((1, 0, 2))
    features = features.reshape((features.shape[0], -1))
    logits = self.clf.predict_log_proba(features)
    return ch.from_numpy(logits).cuda()


class PyTorchWrapper:
  def __init__(self, model):
    self.model = model

  def __call__(self, x):
    return model(x)


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module # that I actually define.
    def forward(self, x, with_latent=False):
        return self.module(x, with_latent=with_latent)


def finetune_into_binary(m, last_binary=True, on_cpu=False):
    # Freeze feature extraction layers
    for n, l in m.named_parameters():
        if "features." in n:
            l.requires_grad  = False
    # Swap out final classification layer
    if last_binary:
        m.classifier[6] = ch.nn.Linear(m.classifier[6].weight.shape[1], 1)
    if on_cpu:
        return m
    return ch.nn.DataParallel(m.cuda())


def finetune_into_binary_with_features(m, num_latent, on_cpu=False):
    # Freeze feature extraction layers
    for n, l in m.named_parameters():
        if "features." in n:
            l.requires_grad  = False
    new_model = VggFeatureExtractor(m, num_latent)
    if on_cpu:
        return new_model
    return ch.nn.DataParallel(new_model.cuda())


def get_cifar_dataloaders(batch_size=256):
    # Construct artificial CIFAR10 car/bird data
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(.25,.25,.25),
                                            transforms.RandomRotation(2),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    
    transform_validation = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    trainset = torchvision.datasets.CIFAR10(root='/p/adversarialml/as9rw/datasets/cifar10', train=True, download=True, transform=transform_train)
    trainloader = ch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='/p/adversarialml/as9rw/datasets/cifar10', train=False, download=True, transform=transform_validation)
    testloader = ch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader


def make_folder_loaders(path):
    transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(.25,.25,.25),
                                          transforms.RandomRotation(2),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    
    transform_validation = transforms.Compose([transforms.Resize((32, 32)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    
    trainset = torchvision.datasets.ImageFolder(root=os.path.join(path, "train"), transform=transform_train)
    testset  = torchvision.datasets.ImageFolder(root=os.path.join(path, "test"),   transform=transform_validation)
    
    trainloader = ch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,  num_workers=4)
    testloader  = ch.utils.data.DataLoader(testset,  batch_size=128, shuffle=False, num_workers=4)
    return trainloader, testloader


def get_multiclass_acc(model, data_loader):
    acc, count = 0, 0
    for im, label in data_loader:
        im, label = im.cuda(), label.cuda()
        with ch.no_grad():
            prediction = model(im)
        acc += ch.sum(ch.argmax(prediction, 1) == label.data).item()
        count += im.shape[0]
    return acc / count
