# code from here - https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb

import torchvision.transforms as T
from PIL import Image

# Indoor360 classes
CLASSES = [
  'toilet', 'board', 'mirror', 'bed', 'potted plant',
  'book', 'clock', 'phone', 'keyboard', 'tv',
  'fan', 'backpack', 'light', 'refrigerator', 'bathtub',
  'wine glass', 'airconditioner', 'cabinet', 'sofa', 'bowl',
  'sink', 'computer', 'cup', 'bottle', 'washer',
  'chair', 'picture', 'window', 'door', 'heater',
  'fireplace', 'mouse', 'oven', 'microwave', 'person',
  'vase', 'table'
]


def box_cxcywh_to_xyxy(x):
  x_c, y_c, w, h = x.unbind(1)
  b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
       (x_c + 0.5 * w), (y_c + 0.5 * h)]
  return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
  img_w, img_h = size
  b = box_cxcywh_to_xyxy(out_bbox)
  b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
  return b


def detect(im, model, transform):
  # mean-std normalize the input image (batch-size: 1)
  img = transform(im).unsqueeze(0)

  # demo model only support by default images with aspect ratio between 0.5 and 2
  # if you want to use images with an aspect ratio outside this range
  # rescale your image so that the maximum size is at most 1333 for best results
  assert img.shape[-2] <= 1600 and img.shape[
    -1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

  # propagate through the model
  outputs = model(img)

  # keep only predictions with 0.7+ confidence
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > 0.7

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
  return probas[keep], bboxes_scaled


def plot_results(pil_img, prob, boxes):
  import matplotlib.pyplot as plt
  plt.figure(figsize=(16, 10))
  plt.imshow(pil_img)
  ax = plt.gca()

  COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
  idx = 0
  for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, color=c, linewidth=2))
    cl = p.argmax()
    text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
    ax.text(xmin, ymin, text, fontsize=12,
            bbox=dict(facecolor='yellow', alpha=0.5))
    idx += 1
  print(idx, "number of objs are detected!")
  plt.axis('off')
  plt.show()


from hubconf import _make_detr

if __name__ == '__main__':
  import torch

  # 여기를 수정해주자
  checkpoint = torch.load("output/checkpoint_tmp.pth", map_location='cpu')["model"]

  model = _make_detr("resnet50", dilation=False, num_classes=37)

  model.load_state_dict(checkpoint, strict=False)
  model.eval()

  transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  # image_name = "35446673812_cd3f42325d_o.jpg"  # 1, 13 objs
  image_name = "7l6sG.jpg"  # 2, 22 objs
  image = 'data/indoor360/mollweide_960/' + image_name
  img = Image.open(image)

  pred_logits, pred_boxes = detect(img, model, transform)
  plot_results(img, pred_logits, pred_boxes)
