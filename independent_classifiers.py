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
            # ch.save(best_model, os.path.join("./concept_classifiers", params['description'] + '.pt'))
            ch.save(best_model, os.path.join("./concept_classifiers_small", params['description'] + '.pt'))
    
    final_accuracy = test_model(model, params, mapping)[0]
    writer.add_text('Final_Accuracy', str(final_accuracy), 0)
    writer.close()


if __name__ == "__main__":
    import sys
    basepath   = sys.argv[1]
    mode       = sys.argv[2]
    num_latent = 80
    if mode not in ['test', 'train']:
        raise ValueError("Specify proper mode : train/test")

    clss = basepath.rstrip().split(os.path.sep)[-2]
    
    classes = os.listdir(os.path.join(basepath, "train"))
    concept_classifiers = {x:None for x in classes}

    data_params = {'path': basepath, 'batch_size': 128}
    train_loader, validation_loader = utils.make_folder_loaders(basepath)

    for concept_class in classes:
        train_params = {'description':  clss + "_" + concept_class,
                        'num_epochs': 20, 'check_point': 5,
                        'train_loader': train_loader,
                        'validation_loader': validation_loader}

        if num_latent == 4096:
            concept_classifiers[concept_class] = utils.finetune_into_binary(vgg_model.vgg19_bn(pretrained=True))
        else:
            concept_classifiers[concept_class] = utils.finetune_into_binary_with_features(vgg_model.vgg19_bn(pretrained=True), num_latent=num_latent)
        concept_mapping = {i:0 for i in range(len(classes))}
        concept_mapping[train_loader.dataset.class_to_idx[concept_class]] = 1

        if mode == 'train':
            finetune_model(concept_classifiers[concept_class], train_params, concept_mapping)
        else:
            # Load model
            models_path = sys.argv[3]
            concept_classifiers[concept_class].load_state_dict(ch.load(os.path.join(models_path, train_params['description'] + ".pt")))
            concept_classifiers[concept_class].eval()
            # Calculate F-1 metrics
            gran = 20
            best_f1 = 0
            for i in range(gran + 1):
                acc = test_model(concept_classifiers[concept_class], train_params, concept_mapping, i / gran)
                if acc[1] > best_f1:
                    best_f1 = acc[1]
            print("%s : best F-1 Score : %.3f" % (train_params['description'], best_f1))
            # Free up space
            del concept_classifiers[concept_class]
