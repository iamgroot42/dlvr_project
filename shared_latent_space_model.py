import torch as ch
import torch.nn as nn
import os
import vgg_model
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import copy
from tqdm import tqdm
import numpy as np


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



def get_cifar_dataloaders():
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
    trainloader = ch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='/p/adversarialml/as9rw/datasets/cifar10', train=False, download=True, transform=transform_validation)
    testloader = ch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader


def make_small_concept_model(dim_in):
    model = nn.Sequential(nn.Linear(dim_in, 32), nn.ReLU(True),
                           nn.Linear(32, 8), nn.ReLU(True),
                           nn.Linear(8, 1))
    return model.cuda()


def batch_cycle_repeater(dloader):
    loader = iter(dloader)
    while True:
        try:
            image, label = loader.next()
        except StopIteration:
            loader = iter(dloader)
            image, label = loader.next()
        yield (image, label)


def train_shared_latent_model(model, nb_epochs, cifar_loaders, concept_loaders, concept_models, mappings, latent_ranges, alpha=2):
    writer = SummaryWriter('runs/shared_latent_space_model')
    optimizer = optim.Adam(model.parameters())
    
    # Define loss for main task
    criterion = nn.CrossEntropyLoss()
    # Define loss for inner criterion
    criterion_concept = nn.BCEWithLogitsLoss()
    best_model = copy.deepcopy(model.state_dict())

    # Set model to training mode
    model.train()
        
    for epoch in range(nb_epochs):
        # Each epoch has a training and validation phase
        print("Epoch : %d" % (epoch + 1))
        examples_so_far = 0
        examples_so_far_inside = 0
            
        # Loss accumulator for each epoch
        logs = {'Loss': 0.0, 'Accuracy': 0.0, 'Aux_Loss': 0.0}
            
        # Iterate over training data
        iterator = tqdm(cifar_loaders[0])
        for image, label in iterator:
            image = image.cuda()
            label = label.cuda()

            # Zero gradient
            optimizer.zero_grad()

            # Forward pass
            prediction = model(image)
            loss = criterion(prediction, label)

            accuracy = ch.sum(ch.argmax(prediction, 1) == label.data).item()
                    
            with ch.no_grad():
                y_ = (ch.argmax(prediction, 1)).flatten().cpu().numpy() * 1

            # Update log
            examples_so_far += image.shape[0]
            logs['Loss'] += image.shape[0]*loss.detach().item()
            logs['Accuracy'] += accuracy

            # BackProp
            loss.backward()
            optimizer.step()

            # Repeat for alpha randomly-sampled concept classifiers
            random_concepts = np.random.choice(len(concept_loaders), alpha)
            for i in random_concepts:
                # Pick random concept to focus on
                focus_concept = np.random.choice(mappings[i])
                image, label = next(concept_loaders[i][0])
                image = image.cuda()
                label = label.cuda()

                # Convert labels using the given mapping
                for j, l in enumerate(label):
                    label[j] = (l.item() == focus_concept) * 1

                # Zero gradient
                optimizer.zero_grad()

                # Forward pass
                latent, _ = model(image, with_latent=True)

                # Filter out relevant latent code
                latent_code = latent[:, latent_ranges[i][focus_concept][0]: latent_ranges[i][focus_concept][1]]

                # Pass through corresponding concept classifier
                prediction = concept_models[i][focus_concept](latent_code)
                loss = criterion_concept(prediction, label.unsqueeze(1).float())

                # Update log
                examples_so_far_inside += image.shape[0]
                logs['Aux_Loss'] += loss.detach().item() * image.shape[0]

                # BackProp
                loss.backward()
                optimizer.step()

            # Update logging stats
            iterator.set_description('Aux Loss: %.3f Loss: %.3f Accuracy: %.3f' % (logs['Aux_Loss'] / examples_so_far_inside,
                                                                                   logs['Loss'] / examples_so_far,
                                                                                   100 * logs['Accuracy'] / examples_so_far))
            
        # Normalize and write the data to TensorBoard
        logs['Loss']      /= examples_so_far
        logs['Accuracy']  /= examples_so_far
            
        writer.add_scalars('Loss', {"train": logs['Loss']}, epoch)
        writer.add_scalars('Accuracy', {"train": logs['Accuracy']}, epoch)
            
        print("%s loss: %.3f , accuracy: %.3f" % ("train", logs['Loss'], logs['Accuracy']))
                 
    #     # Write best weights to disk
    #     if epoch % 5 == 0 or epoch == nb_epochs - 1:
    #         ch.save(best_model, "./shared_latent_space_model.pt")
    
    # final_accuracy = test_model(model, params)
    # writer.add_text('Final_Accuracy', str(final_accuracy), 0)
    # writer.close()


if __name__ =="__main__":
    import sys
    concepts_folder = sys.argv[1]
    # Architecture specifics
    per_class_concept_latent = 80
    num_total_concepts       = 48
    # Create class-wise concept data loaders
    start_index = 0
    concept_loaders = []
    latent_ranges   = []
    mappings        = []
    concept_models  = []
    # Define specifics for training
    for class_folder in os.listdir(concepts_folder):
        concept_loader = make_folder_loaders(os.path.join(concepts_folder, class_folder))
        # Create mappings
        num_concepts = len(concept_loader[0].dataset.classes)
        mappings.append(num_concepts)
        concept_loaders.append((batch_cycle_repeater(concept_loader[0]), concept_loader[1]))
        # Create concept models
        concept_models.append([make_small_concept_model(per_class_concept_latent) for _ in range(num_concepts)])
        # Create latent ranges
        latent_range = []
        for i in range(num_concepts):
            latent_range.append((start_index, start_index + per_class_concept_latent))
            start_index += per_class_concept_latent
        latent_ranges.append(latent_range)
    # Creaate normal CIFAR-10 loader
    cifar_loaders = get_cifar_dataloaders()
    # Define model
    model = nn.DataParallel(vgg_model.vgg19_bn(pretrained=False, num_latent=per_class_concept_latent * num_total_concepts))
    #  Train model with shared encoders
    nb_epochs = 100
    train_shared_latent_model(model, nb_epochs,
                              cifar_loaders, concept_loaders,
                              concept_models, mappings,
                              latent_ranges, alpha=2)
