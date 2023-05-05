import torch
import torchvision.models as models
edsr = models.edsr(pretrained=True).eval()

img = cv2.imread('images/000000000435.tif')
img = torch.from_numpy(img.transpose((2, 0, 1))).float().div(255)
with torch.no_grad():
    result = edsr(img.unsqueeze(0)).squeeze(0)
result = (result.mul(255).clamp(0, 255).round().cpu().numpy().transpose((1, 2, 0))).astype('uint8')
cv2.imwrite('output.tif', result)
#writes the output as "result" window