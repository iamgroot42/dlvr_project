import torch as ch


def finetune_into_binary(m, last_binary=True):
    # Freeze feature extraction layers
    for n, l in m.named_parameters():
        if "features." in n:
            l.requires_grad  = False
    # Swap out final classification layer
    if last_binary:
    	m.classifier[6] = ch.nn.Linear(m.classifier[6].weight.shape[1], 1)
    return ch.nn.DataParallel(m.cuda())
