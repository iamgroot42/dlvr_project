import torch as ch
from sklearn.ensemble import RandomForestClassifier
import torchvision
import vgg_model
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import utils


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


def get_combined_data(loader):
	X, Y = [], []
	for x,y in tqdm(loader):
		X.append(x)
		Y.append(y)
	return (ch.cat(X, 0), ch.cat(Y, 0))


def get_actual_scores(X, classifier, batch_size=5000):
	features = []
	i = 0
	while i < len(X):
		with ch.no_grad():
			score = classifier(X[i:i + batch_size]).cpu()
		features.append(score)
		i += batch_size
	return ch.cat(features, 0)


if __name__ =="__main__":
	import sys
	cc_dir = sys.argv[1]
	# Get CIFAR10 loaders
	trainloader, testloader = get_cifar_dataloaders()
	X_train, Y_train = get_combined_data(trainloader)
	X_val,   Y_val  = get_combined_data(testloader)
	# Shift labels to cpu,numpy
	Y_train = Y_train.cpu().numpy()
	Y_val   = Y_val.cpu().numpy()
	# Use concept classifiers to get scores
	features_train, features_test = [], []
	for ccpath in tqdm(os.listdir(cc_dir)):
		model = utils.finetune_into_binary(vgg_model.vgg19_bn(pretrained=True))
		# Load weights into model
		model.load_state_dict(ch.load(os.path.join(cc_dir, ccpath)))
		# Set to evaluation mode
		model.eval()
		features_train.append(get_actual_scores(X_train, model))
		features_test.append(get_actual_scores(X_val, model))
		# Explicitly free memory
		del model
	features_train = ch.stack(features_train).squeeze(-1).numpy().transpose()
	features_test  = ch.stack(features_test).squeeze(-1).numpy().transpose()
	
	# Train RFC using these fratures
	clf = RandomForestClassifier(max_depth=5, random_state=0)
	# Train model
	clf.fit(features_train, Y_train)
	# Display performance on training data
	print("Accuracy on train data : %.4f" % (100 * clf.score(features_train, Y_train)))
	print("Accuracy on test data : %.4f"  % (100 * clf.score(features_test, Y_val)))
	