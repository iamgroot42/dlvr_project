import torch as ch
import numpy as np
import torch.nn as nn
import itertools
from kornia.geometry.transform import resize, rotate, center_crop, translate
from tqdm import tqdm
import os
import pickle

import vgg_model, utils


def _apply_black_border(x, border_size):
  orig_height = x.shape[2]
  orig_width  = x.shape[3]
  x = resize(x, (orig_width - 2*border_size,
                                 orig_height - 2*border_size))
  return nn.ConstantPad2d(border_size, 0)(x)


def apply_transformation(x, trans, on_cpu):
  dx, dy, angle = trans[0], trans[1], trans[2]
  height, width = x.shape[2], x.shape[3]

  # Pad the image to prevent two-step rotation / translation from truncating
  # corners
  max_dist_from_center = np.sqrt(height**2+width**2) / 2
  min_edge_from_center = float(np.min([height, width])) / 2
  padding = np.ceil(max_dist_from_center -
                    min_edge_from_center).astype(np.int32)
  x = nn.ConstantPad2d(padding, 0)(x)

  # Apply rotation
  angle = ch.from_numpy(np.ones(x.shape[0]) * angle)
  if not on_cpu:
    angle = angle.to(x.get_device())
  x = rotate(x, angle)

  # Apply translation
  dx_in_px = -dx * height
  dy_in_px = -dy * width
  translation = ch.from_numpy(np.tile(np.array([dx_in_px, dy_in_px], dtype=np.float32), (x.shape[0], 1)))
  if not on_cpu:
    translation = translation.to(x.get_device())
  x = translate(x, translation)
  x = translate(x, translation)
  # Pad if needed
  if x.shape[2] < height or x.shape[3] < width:
    pad = nn.ConstantPad2d((0, max(0, height - x.shape[2]), 0, max(0, width - x.shape[3])), 0)
    x = pad(x)
  return center_crop(x, (height, width))


def spatial_transformation_method(model, x, n_samples=None,
                                  dx_min=-0.1, dx_max=0.1, n_dxs=2,
                                  dy_min=-0.1, dy_max=0.1, n_dys=2,
                                  angle_min=-30, angle_max=30, n_angles=6,
                                  black_border_size=0, on_cpu=False):
  if dx_min < -1 or dy_min < -1 or dx_max > 1 or dy_max > 1:
      raise ValueError("The value of translation must be bounded "
                       "within [-1, 1]")
  
  y = ch.argmax(model(x), 1)

  # Define the range of transformations
  dxs = np.linspace(dx_min, dx_max, n_dxs)
  dys = np.linspace(dy_min, dy_max, n_dys)
  angles = np.linspace(angle_min, angle_max, n_angles)

  if n_samples is None:
    transforms = list(itertools.product(*[dxs, dys, angles]))
  else:
    sampled_dxs = np.random.choice(dxs, n_samples)
    sampled_dys = np.random.choice(dys, n_samples)
    sampled_angles = np.random.choice(angles, n_samples)
    transforms = zip(sampled_dxs, sampled_dys, sampled_angles)

  x = _apply_black_border(x, black_border_size)
  transformed_ims = []
  for transform in transforms:
    transformed_ims.append(apply_transformation(x, transform, on_cpu))
  transformed_ims = ch.stack(transformed_ims)

  def _compute_xent(x):
    logits = model(x)
    loss = nn.CrossEntropyLoss(reduction='none')
    return loss(logits, y)

  all_xents = ch.stack([_compute_xent(x) for x in transformed_ims])
  
  # We want the worst case sample, with the largest xent_loss
  worst_sample_idx = ch.argmax(all_xents, 0)  # B
  transformed_ims_perm = ch.transpose(transformed_ims, 0, 1)
  after_lookup = []
  for i, sample_id in enumerate(worst_sample_idx):
    after_lookup.append(transformed_ims_perm[i][sample_id])
  after_lookup = ch.stack(after_lookup)

  return after_lookup



if __name__ == "__main__":
  import sys
  model_path = sys.argv[1]
  _, val_loader = utils.get_cifar_dataloaders(64)
  iterator = tqdm(val_loader)
  per_class_concept_latent = 80
  num_total_concepts       = 48
  on_cpu = True
  # C3, C4, Standard, Robust
  # Load model
  # model = vgg_model.vgg19_bn(pretrained=False, num_latent=per_class_concept_latent * num_total_concepts).cuda()
  # checkpoint = ch.load(model_path)
  # model.load_state_dict(checkpoint.module.state_dict())
  # model.eval()
  # wrapped_model = utils.PyTorchWrapper(model)
  # C1, C2
  latent = True
  models = []
  if latent:
    filename = "./meta_classifier_True"
  else:
    filename = "./meta_classifier_False"
  # Load meta-classifier
  clf = pickle.load(open(filename, 'rb'))

  print("[Concept Classifiers] Loading")
  for ccpath in os.listdir(model_path):
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
  print("[Concept Classifiers] Loaded")
  acc, n_samples = 0, 0
  for (im, label) in iterator:
    n_samples += im.shape[0]
    if not on_cpu:
      im = im.cuda()
    adv_x = spatial_transformation_method(wrapped_model, im, on_cpu=on_cpu)
    preds = ch.argmax(wrapped_model(adv_x), 1).cpu()
    acc  += ch.sum(label == preds).cpu().item()
    iterator.set_description('Accuracy : %.2f' % (100 * acc / n_samples))
