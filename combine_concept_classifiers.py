import torch as ch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from poutyne.framework import Model
import torchvision
import vgg_model
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import os
import utils


def get_combined_data(loader):
	X, Y = [], []
	for x,y in tqdm(loader):
		X.append(x)
		Y.append(y)
	return (ch.cat(X, 0), ch.cat(Y, 0))


def get_actual_scores(X, classifier, latent, batch_size=5000):
	features = []
	i = 0
	while i < len(X):
		with ch.no_grad():
			if latent:
				score = classifier(X[i:i + batch_size]).cpu()
			else:
				score = classifier(X[i:i + batch_size], with_latent=True)[0].cpu()
		features.append(score)
		i += batch_size
	return ch.cat(features, 0)


if __name__ =="__main__":
	import sys
	cc_dir           = sys.argv[1]
	combination_mode = int(sys.argv[2])
	if combination_mode == 1:
		latent = False
	else:
		latent = True
	# Get CIFAR10 loaders
	trainloader, testloader = utils.get_cifar_dataloaders()
	X_train, Y_train = get_combined_data(trainloader)
	X_val,   Y_val   = get_combined_data(testloader)
	# Shift labels to cpu,numpy
	Y_train = Y_train.cpu().numpy()
	Y_val   = Y_val.cpu().numpy()
	# Use concept classifiers to get scores
	features_train, features_test = [], []
	for ccpath in tqdm(os.listdir(cc_dir)):
		if latent:
			model = utils.finetune_into_binary_with_features(vgg_model.vgg19_bn(pretrained=True), num_latent=80)
		else:
			model = utils.finetune_into_binary(vgg_model.vgg19_bn(pretrained=True))
		# Load weights into model
		model.load_state_dict(ch.load(os.path.join(cc_dir, ccpath)))
		# Set to evaluation mode
		model.eval()
		features_train.append(get_actual_scores(X_train, model, latent))
		features_test.append(get_actual_scores(X_val, model, latent))
		# Explicitly free memory
		del model

	features_train = ch.stack(features_train).squeeze(-1).numpy().transpose()
	features_test  = ch.stack(features_test).squeeze(-1).numpy().transpose()

	if len(features_train.shape) == 3:
		features_train = np.transpose(features_train, (1, 0, 2))
		features_test  = np.transpose(features_test, (1, 0, 2))
		features_train = np.reshape(features_train, (features_train.shape[0], -1))
		features_test  = np.reshape(features_test, (features_test.shape[0], -1))

	if latent:
		# Train RFC using these features
		clf = MLPClassifier(hidden_layer_sizes=(512, 128, 32))
	else:
		# Train NN using these features
		clf = RandomForestClassifier(max_depth=5, random_state=0)
	# Train model
	clf.fit(features_train, Y_train)
	# Display performance on training data
	print("Accuracy on train data : %.4f" % (100 * clf.score(features_train, Y_train)))
	print("Accuracy on test  data : %.4f"  % (100 * clf.score(features_test, Y_val)))
	