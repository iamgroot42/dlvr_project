import os
from PIL import Image
import vgg_model
import torch as ch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm as pbar
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import copy

import utils


def test_model(model, params):
    phase = 'validation'
    logs = {'Accuracy': 0.0}
            
    preds, gt = [], []
    # Iterate over data
    for image, label in params[phase+'_loader']:
        image = image.cuda()

        with ch.no_grad():
            prediction = ch.argmax(model(image), 1).cpu()
            for pred in prediction:
                preds.append(pred)

    logs['Accuracy'] = accuracy_score(gt, preds)
    
    return logs['Accuracy']


def train_model(model, params):
    writer = SummaryWriter('runs/' + params['description'])
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    total_updates = params['num_epochs'] * len(params['train_loader'])
    
    criterion = ch.nn.CrossEntropyLoss()
    best_acc = test_model(model, params)
    best_model = copy.deepcopy(model.state_dict())
        
    for epoch in range(params['num_epochs']):
        # Each epoch has a training and validation phase
        print("Epoch : %d" % (epoch + 1))
        for phase in ['train', 'validation']:
            examples_so_far = 0
            
            # Loss accumulator for each epoch
            logs = {'Loss': 0.0,
                    'Accuracy': 0.0}
            
            # Set the model to the correct phase
            model.train() if phase == 'train' else model.eval()
            
            # Iterate over data
            for image, label in params[phase+'_loader']:
                image = image.cuda()
                label = label.cuda()

                # Zero gradient
                optimizer.zero_grad()

                with ch.set_grad_enabled(phase == 'train'):
                    
                    # Forward pass
                    prediction = model(image)
                    loss = criterion(prediction, label.unsqueeze(1).float())
                    accuracy = ch.sum(ch.argmax(prediction, 1) == label.data.unsqueeze(1)).item()
                    
                    with ch.no_grad():
                        y_ = (ch.argmax(prediction, 1).flatten().cpu().numpy()) * 1

                    # Update log
                    examples_so_far += image.shape[0]
                    logs['Loss'] += image.shape[0]*loss.detach().item()
                    logs['Accuracy'] += accuracy

                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            
            # Normalize and write the data to TensorBoard
            logs['Loss']      /= examples_so_far
            logs['Accuracy']  /= examples_so_far
            
            writer.add_scalars('Loss', {phase: logs['Loss']}, epoch)
            writer.add_scalars('Accuracy', {phase: logs['Accuracy']}, epoch)
            
            print("%s loss: %.3f , accuracy: %.3f" % (phase, logs['Loss'], logs['Accuracy']))

            # Save the best weights
            if phase == 'validation' and logs['Accuracy'] > best_acc:
                best_acc = logs['Accuracy']
                best_model = copy.deepcopy(model.state_dict())
                 
        # Write best weights to disk
        if epoch % params['check_point'] == 0 or epoch == params['num_epochs']-1:
            ch.save(best_model, "./use_all_data.pt")
    
    final_accuracy = test_model(model, params, mapping)
    writer.add_text('Final_Accuracy', str(final_accuracy), 0)
    writer.close()


def make_folder_loaders(params):
    transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(.25,.25,.25),
                                          transforms.RandomRotation(2),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    
    transform_validation = transforms.Compose([transforms.Resize((32, 32)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    
    trainset = torchvision.datasets.ImageFolder(root=os.path.join(params['path'], "train"), transform=transform_train)
    testset  = torchvision.datasets.ImageFolder(root=os.path.join(params['path'], "test"),   transform=transform_validation)
    
    trainloader = ch.utils.data.DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, num_workers=4)
    testloader  = ch.utils.data.DataLoader(testset, batch_size=params['batch_size'], shuffle=False, num_workers=4)
    return trainloader, testloader


if __name__ == "__main__":
    import sys
    datapath = sys.argv[1]
    
    data_params = {'path': datapath, 'batch_size': 256}
    train_loader, validation_loader = make_folder_loaders(data_params)

    train_params = {'description':  "use_all_data",
                    'num_epochs': 100, 'check_point': 5,
                    'train_loader': train_loader,
                    'validation_loader': validation_loader}

    classifier = utils.finetune_into_binary(vgg_model.vgg19_bn(pretrained=True), False)
    train_model(classifier, train_params)
