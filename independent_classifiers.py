from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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



def test_model(model, params, mapping, threshold=0.5):
    phase = 'validation'
    logs = {'Accuracy': 0.0, 'F1-Score': 0.0}
            
    preds, gt = [], []
    # Iterate over data
    for image, label in params[phase+'_loader']:
        image = image.cuda()
        # Convert labels using the given mapping
        for i, l in enumerate(label):
            gt.append(mapping[l.item()])

        with ch.no_grad():
            prediction = ch.sigmoid(model(image)).cpu()
            for pred in prediction:
                preds.append(pred >= threshold)

    logs['Accuracy'] = accuracy_score(gt, preds)
    logs['F1-Score'] = f1_score(gt, preds)
    
    return logs['Accuracy'], logs['F1-Score']


def finetune_model(model, params, mapping, threshold=0.3):
    writer = SummaryWriter('runs/' + params['description'])
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    total_updates = params['num_epochs'] * len(params['train_loader'])
    
    criterion = ch.nn.BCEWithLogitsLoss()
    best_f1 = test_model(model, params, mapping)[1]
    best_model = copy.deepcopy(model.state_dict())
        
    for epoch in range(params['num_epochs']):
        # Each epoch has a training and validation phase
        print("Epoch : %d" % (epoch + 1))
        for phase in ['train', 'validation']:
            examples_so_far = 0
            
            # Loss accumulator for each epoch
            logs = {'Loss': 0.0,
                    'Accuracy': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0}
            
            # Set the model to the correct phase
            model.train() if phase == 'train' else model.eval()
            
            # Iterate over data
            for image, label in params[phase+'_loader']:
                # Convert labels using the given mapping
                for i, l in enumerate(label):
                    label[i] = mapping[l.item()]
                
                image = image.cuda()
                label = label.cuda()

                # Zero gradient
                optimizer.zero_grad()

                with ch.set_grad_enabled(phase == 'train'):
                    
                    # Forward pass
                    prediction = model(image)
                    loss = criterion(prediction, label.unsqueeze(1).float())

                    accuracy = ch.sum((ch.sigmoid(prediction) >= threshold) == label.data.unsqueeze(1)).item()
                    
                    with ch.no_grad():
                        y_ = ((ch.sigmoid(prediction)).flatten().cpu().numpy() >= threshold) * 1

                    # Update log
                    examples_so_far += image.shape[0]
                    logs['Loss'] += image.shape[0]*loss.detach().item()
                    logs['Accuracy'] += accuracy
                    logs['Precision'] += image.shape[0]*precision_score(label.data.cpu(), y_)
                    logs['Recall'] += image.shape[0]*recall_score(label.data.cpu(), y_)

                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            
            # Normalize and write the data to TensorBoard
            logs['Loss']      /= examples_so_far
            logs['Accuracy']  /= examples_so_far
            logs['Precision'] /= examples_so_far
            logs['Recall']    /= examples_so_far
            
            f1_score = (2 * logs['Precision'] * logs['Recall']) / (logs['Precision'] + logs['Recall'])
            writer.add_scalars('Loss', {phase: logs['Loss']}, epoch)
            writer.add_scalars('Accuracy', {phase: logs['Accuracy']}, epoch)
            writer.add_scalars('F-1', {phase: f1_score}, epoch)
            
            print("%s loss: %.3f , accuracy: %.3f, F-1: %.3f" % (phase, logs['Loss'], logs['Accuracy'], f1_score))

            # Save the best weights
            if phase == 'validation' and f1_score > best_f1:
                best_f1 = f1_score
                best_model = copy.deepcopy(model.state_dict())
                 
        # Write best weights to disk
        if epoch % params['check_point'] == 0 or epoch == params['num_epochs']-1:
            ch.save(best_model, os.path.join("./concept_classifiers", params['description'] + '.pt'))
    
    final_accuracy = test_model(model, params, mapping)[0]
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
    basepath = sys.argv[1]
    clss = basepath.rstrip().split(os.path.sep)[-2]
    
    classes = os.listdir(os.path.join(basepath, "train"))
    concept_classifiers = {x:None for x in classes}

    data_params = {'path': basepath, 'batch_size': 128}
    train_loader, validation_loader = make_folder_loaders(data_params)

    print(train_loader.dataset.classes)
    print(train_loader.dataset.class_to_idx)
    print(validation_loader.dataset.classes)
    print(validation_loader.dataset.class_to_idx)

    for concept_class in classes:
        train_params = {'description':  clss + "_" + concept_class,
                        'num_epochs': 20, 'check_point': 5,
                        'train_loader': train_loader,
                        'validation_loader': validation_loader}

        concept_classifiers[concept_class] = utils.finetune_into_binary(vgg_model.vgg19_bn(pretrained=True))
        concept_mapping = {i:0 for i in range(len(classes))}
        concept_mapping[train_loader.dataset.class_to_idx[concept_class]] = 1
        finetune_model(concept_classifiers[concept_class], train_params, concept_mapping)

        for i in range(11):
            acc = test_model(concept_classifiers[concept_class], train_params, concept_mapping, i / 10)
            print("Threshold %.2f : Acc %.3f, F-1 %.3f" % (i/10, acc[0], acc[1]))
