from cleverhans.future.torch.attacks import projected_gradient_descent
from hsja import hop_skip_jump_attack
import torch as ch
import numpy as np
from tqdm import tqdm
import pickle
import os

import vgg_model, utils



if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    is_latent  = int(sys.argv[2])
    if is_latent == 1:
        latent = False
    else:
        latent = True
    # Load model
    per_class_concept_latent = 80
    num_total_concepts       = 48
    gpu_devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    # C3
    if is_latent == 3:
        wrapped_model = vgg_model.vgg19_bn(pretrained=False, num_latent=per_class_concept_latent * num_total_concepts).cuda()
        checkpoint = ch.load(model_path)
        wrapped_model.load_state_dict(checkpoint.module.state_dict())
        wrapped_model.eval()
    # C1, C2
    else:
        # Load meta-classifier
        models = []
        if latent:
            filename = "./meta_classifier_True"
        else:
            filename = "./meta_classifier_False"
        multi_gpus = True
        clf = pickle.load(open(filename, 'rb'))
        for i, ccpath in tqdm(enumerate(os.listdir(model_path))):
            if latent:
                model = utils.finetune_into_binary_with_features(vgg_model.vgg19_bn(pretrained=True), num_latent=80, on_cpu=multi_gpus)
            else:
                model = utils.finetune_into_binary(vgg_model.vgg19_bn(pretrained=True), on_cpu=multi_gpus)
            # Load weights into model
            if multi_gpus:
                # model.load_state_dict(ch.load(os.path.join(model_path, ccpath), map_location='cpu'))
                model = utils.WrappedModel(model)
                checkpoint = ch.load(os.path.join(model_path, ccpath), map_location=gpu_devices[i % len(gpu_devices)])
                model = model.to(gpu_devices[i % len(gpu_devices)])
                model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(ch.load(os.path.join(model_path, ccpath)))
            # Set to evaluation mode
            model.eval()
            models.append(model)
        wrapped_model = utils.MultipleModelsWrapper(models, clf, latent)
        
    # Run attacks
    _, val_loader = utils.get_cifar_dataloaders(2)
    misclass, total = 0, 0
    iterator = tqdm(val_loader)

    eps = 8/255
    nb_steps = 20
    eps_iter = eps * 2.5 / nb_steps

    for (im, label) in iterator:
        # Boundary++ Attack
        im = im.cuda()
        advs = hop_skip_jump_attack(wrapped_model, im, np.inf, verbose=False)
        # PGD Attack
        # advs = projected_gradient_descent(model, im, eps=eps, eps_iter=eps_iter, nb_iter=nb_steps, norm=np.inf)
        pert_labels = ch.argmax(wrapped_model(advs), 1).cpu()

        misclass += (pert_labels != label).sum().item()
        total    += len(label)

        iterator.set_description('Attack success rate : %.2f' % (100 * misclass / total))
