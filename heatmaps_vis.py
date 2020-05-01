import torch as ch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

import utils, vgg_model


def norm_grad(grad):
	grad_normed = grad
	for i, g in enumerate(grad_normed): 
		grad_normed[i] = (g - g.min())
		grad_normed[i] /= grad_normed[i].max()
	return grad_normed


if __name__ == "__main__":
	custom = False
	if custom:
		import sys
		model_path = sys.argv[1]
		# Define and load trained model
		model = vgg_model.vgg19_bn(pretrained=False, num_latent=80 * 48).cuda()
		checkpoint = ch.load(model_path)
		model.load_state_dict(checkpoint.module.state_dict())
	else:
		# Standard model
		model = vgg_model.vgg19_bn(pretrained=False).cuda()
	model.eval()
	_, val_loader = utils.get_cifar_dataloaders(16)
	im, label = iter(val_loader).next()
	im = im.cuda()

	label = label.cuda()
	loss = ch.nn.CrossEntropyLoss()
	im.requires_grad_()

	if custom:
		focus_concept = 15
		lrange = (focus_concept * 80, (focus_concept + 1) * 80)
		latent, _ = model(im, with_latent=True)

		mask = ch.zeros_like(latent)
		for i in range(mask.shape[0]):
			mask[i][lrange[0]:lrange[1]] = 1

		scores = model.classifier[-1](latent * mask)

	else:
		scores = model(im)

	loss_value = loss(scores, label)
	grad = ch.autograd.grad(loss_value.mean(), [im])[0].cpu()
	saliency = norm_grad(grad)

	index = 8
	raw_image   = im[index].cpu().detach().numpy()
	raw_image -= raw_image.min()
	raw_image /= raw_image.max()
	raw_heatmap = saliency[index].cpu().numpy()
	# plt.imshow(raw_image.transpose(1, 2, 0))
	plt.imshow(raw_heatmap.transpose(1, 2, 0))
	plt.axis('off')
	plt.savefig("saliency_map.png")
