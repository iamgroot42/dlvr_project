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
    # Load model
    per_class_concept_latent = 80
    num_total_concepts       = 48
    # C3
    # model = vgg_model.vgg19_bn(pretrained=False, num_latent=per_class_concept_latent * num_total_concepts).cuda()
    # checkpoint = ch.load(model_path)
    # model.load_state_dict(checkpoint.module.state_dict())
    # model.eval()
    # C1, C2
    # Load meta-classifier
    latent = False
    models = []
    if latent:
        filename = "./meta_classifier_True"
    else:
        filename = "./meta_classifier_False"
    on_cpu = True
    clf = pickle.load(open(filename, 'rb'))
    for ccpath in tqdm(os.listdir(model_path)):
        if latent:
            model = utils.finetune_into_binary_with_features(vgg_model.vgg19_bn(pretrained=True), num_latent=80, on_cpu=on_cpu)
        else:
            model = utils.finetune_into_binary(vgg_model.vgg19_bn(pretrained=True), on_cpu=on_cpu)
        # Load weights into model
        if on_cpu:
            model.load_state_dict(ch.load(os.path.join(model_path, ccpath), map_location='cpu'))
        else:
            model.load_state_dict(ch.load(os.path.join(model_path, ccpath)))
        # Set to evaluation mode
        model.eval()
        models.append(model)
    wrapped_model = utils.MultipleModelsWrapper(models, clf, latent)

    # Run attacks
    _, val_loader = utils.get_cifar_dataloaders(16)
    misclass, total = 0, 0
    iterator = tqdm(val_loader)

    eps = 8/255
    nb_steps = 20
    eps_iter = eps * 2.5 / nb_steps

    for (im, label) in iterator:
        # Boundary++ Attack
        if not on_cpu:
            im = im.cuda()
        try:
            advs = hop_skip_jump_attack(model, im, np.inf, verbose=False)
            # PGD Attack
            # advs = projected_gradient_descent(model, im, eps=eps, eps_iter=eps_iter, nb_iter=nb_steps, norm=np.inf)
            pert_labels = ch.argmax(model(advs), 1).cpu()

            misclass += (pert_labels != label).sum().item()
            total    += len(label)
        except:
            # If assert fails, attack may have failed for batch: try another batch
            continue

        iterator.set_description('Attack success rate : %.2f' % (100 * misclass / total))
