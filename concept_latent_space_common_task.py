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


def make_small_concept_model(dim_in):
    model = nn.DataParallel(nn.Sequential(nn.Linear(dim_in, 32), nn.ReLU(True),
                           nn.Linear(32, 8), nn.ReLU(True),
                           nn.Linear(8, 1)))
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


def train_shared_latent_model(model, nb_epochs, cifar_loaders, concept_loaders, mappings, latent_ranges, alpha=2, beta=1):
    writer = SummaryWriter('runs/shared_latent_space_model')
    optimizer = optim.Adam(model.parameters())
    
    # Define loss for main task
    criterion = nn.CrossEntropyLoss()
        
    for epoch in range(nb_epochs):
        # Set model to training mode
        model.train()

        # Each epoch has a training and validation phase
        print("Epoch : %d" % (epoch + 1))
        examples_so_far = 0
            
        # Loss accumulator for each epoch
        logs = {'Loss': 0.0, 'Accuracy': 0.0}
            
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

            # Update log
            examples_so_far += image.shape[0]
            logs['Loss'] += image.shape[0]*loss.detach().item()
            logs['Accuracy'] += accuracy

            # BackProp
            loss.backward()
            optimizer.step()

            if np.random.rand() <= beta:
                # Repeat for alpha randomly-sampled concept classifiers
                random_concepts = np.random.choice(len(concept_loaders), alpha)
                for i in random_concepts:
                    # Pick random concept to focus on
                    focus_concept = np.random.choice(mappings[i])
                    image, label = next(concept_loaders[i][0])
                    image = image.cuda()
                    label = label.cuda()

                    # Zero gradient
                    optimizer.zero_grad()

                    # Forward pass
                    latent, _ = model(image, with_latent=True)

                    # Zero out all other tensors while passing inputs forward
                    latent[:, :latent_ranges[i][focus_concept][0]] = 0
                    latent[:, latent_ranges[i][focus_concept][1]:] = 0

                    # Pass through corresponding concept classifier
                    prediction = model.module.classifier[-1](latent)
                    loss = criterion(prediction, label)

                    accuracy = ch.sum(ch.argmax(prediction, 1) == label.data).item()

                    # Update log
                    examples_so_far += image.shape[0]
                    logs['Loss'] += loss.detach().item() * image.shape[0]
                    logs['Accuracy'] += accuracy

                    # BackProp
                    loss.backward()
                    optimizer.step()

            # Update logging stats
            iterator.set_description('Loss: %.2f Accuracy: %.2f' % (logs['Loss'] / examples_so_far,
                                                                    100 * logs['Accuracy'] / examples_so_far))

            
        # Normalize and write the data to TensorBoard
        logs['Loss']      /= examples_so_far
        logs['Accuracy']  /= examples_so_far
            
        writer.add_scalars('Loss', {"train": logs['Loss']}, epoch)
        writer.add_scalars('Accuracy', {"train": logs['Accuracy']}, epoch)

        # Calculate accuracy metrics on CIFAR-10 validation
        model.eval()
        acc, count = 0, 0
        for im, label in cifar_loaders[1]:
            im, label = im.cuda(), label.cuda()
            with ch.no_grad():
                prediction = model(im)
            acc += ch.sum(ch.argmax(prediction, 1) == label.data).item()
            count += im.shape[0]
        cifar10_val_acc = acc / count

        # Calculate accuracy statistics on all other data loaders
        concept_accs = [[0 for j in range(m)] for m in mappings]
        for i in tqdm(range(len(concept_loaders))):
            concept_count = 0
            # Across all classes
            for im, label in concept_loaders[i][1]:

                im, label = im.cuda(), label.cuda()
                with ch.no_grad():
                    latent, _ = model(im, with_latent=True)
                concept_count += im.shape[0]

                # Across all concepts
                for j in range(mappings[i]):
                    
                    # Zero out all other tensors while passing inputs forward
                    latent[:, :latent_ranges[i][j][0]] = 0
                    latent[:, latent_ranges[i][j][1]:] = 0

                    # Pass through corresponding concept classifier
                    with ch.no_grad():
                        prediction = model.module.classifier[-1](latent)
                    concept_accs[i][j] += ch.sum(ch.argmax(prediction, 1) == label.data).item()

            # Normalize to obtain accuracies
            for j in range(mappings[i]):
                concept_accs[i][j] /= concept_count

        # Flatten out concept accuracies to single list
        concept_accs_flat = []
        for x in concept_accs:
            for y in x:
                concept_accs_flat.append(100 * y)
        # Compute mean, std of concept classifier accuracies
        concept_mean, concept_std = np.mean(concept_accs_flat), np.std(concept_accs_flat)

        # Log statistics on validation data
        print("%s cifar10-acc: %.2f , concepts-acc mean: %.2f , concepts-acc std : %.2f" % ("val", 100 *cifar10_val_acc, concept_mean, concept_std))
                 
        # Write best weights to disk
        if epoch % 5 == 0 or epoch == nb_epochs - 1:
            ch.save(model, "./shared_latent_space_commontask_model_%d_%f.pt" % (alpha, beta))



if __name__ =="__main__":
    import sys
    concepts_folder = sys.argv[1]
    alpha           = int(sys.argv[2])
    beta            = float(sys.argv[3])
    assert 0 <= beta and beta <= 1, "beta should be a valid probability"
    # Architecture specifics
    per_class_concept_latent = 80
    num_total_concepts       = 48
    # Create class-wise concept data loaders
    start_index = 0
    concept_loaders = []
    latent_ranges   = []
    mappings        = []
    # Define specifics for training
    for class_folder in os.listdir(concepts_folder):
        concept_loader = utils.make_folder_loaders(os.path.join(concepts_folder, class_folder))
        # Create mappings
        num_concepts = len(concept_loader[0].dataset.classes)
        mappings.append(num_concepts)
        concept_loaders.append((batch_cycle_repeater(concept_loader[0]), concept_loader[1]))
        # Create latent ranges
        latent_range = []
        for i in range(num_concepts):
            latent_range.append((start_index, start_index + per_class_concept_latent))
            start_index += per_class_concept_latent
        latent_ranges.append(latent_range)
    # Creaate normal CIFAR-10 loader
    cifar_loaders = utils.get_cifar_dataloaders()
    # Define model
    model = nn.DataParallel(vgg_model.vgg19_bn(pretrained=False, num_latent=per_class_concept_latent * num_total_concepts)).cuda()
    #  Train model with shared encoders
    nb_epochs = 150
    train_shared_latent_model(model, nb_epochs,
                              cifar_loaders, concept_loaders,
                              mappings, latent_ranges,
                              alpha=alpha, beta=beta)
