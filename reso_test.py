import cv2
import numpy as np

processed = []
for i in range(len(images)):
    result = cv2.resize(images[i], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    processed.append(denoise_brightness(result))
    for j in label_list[i]: 
        label_list[i][j] = label_list[i][j]* (512/255)

def denoise_brightness(img, original):
    equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side
    # cv2.imwrite('res.png',res)
# cv2.imshow('img', img)
#     cv2.imshow('res',res)

    #h2 is after applying thresholding
    # h2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    #contrast stretched
    # contr = contrast_stretch(res)
    # cv2.imshow('original', original)
    # cv2.imshow('Mean threshold', h2)

    # cv2.imshow('contrast', contr)

    increased = inc(original)
    increased2 = inc2(increased)
    gamma = adjust_gamma(original,1.25)
    new_increased2 = adjust_gamma(increased, 1.25)
    new_increased3 = adjust_gamma(increased2, 1.25)
    # x = concatenate(increased, new_increased1)
    gti = inc2(gamma, 1.25)
    x= np.zeros(img.shape, img.dtype)
    x1 = np.zeros(img.shape, img.dtype)
    x = cv2.addWeighted(gti, 0.4, gamma, 0.6, 0.0, x);
    x1 = cv2.addWeighted(gti, 0.3, gamma, 0.7, 0.0, x1);
    # stretched = contrast_stretch(increased)
    # cv2.imshow('Increased contrast', increased)
    # cv2.imshow('Increased contrast 2', increased2)
    # cv2.imshow('Gamma', gamma)
    # cv2.imwrite('After Gamma.jpg', gamma)
    # cv2.imshow('gama_then_increased', gti)
    # cv2.imshow('New increased2', new_increased2)
    # cv2.imshow('New increased3', new_increased3)
    x2 =Brightness(x1, 2, 15)
    # cv2.imshow('concatenated', x)
    # cv2.imwrite('after concatenation.jpg', x1)
    # cv2.imshow('concatenated2', x1)
    # cv2.imshow('Brighnessed', x2)
    denoised = cv2.fastNlMeansDenoisingColored(x1, None, 8, 10, 7, 21)
    # cv2.imshow('stretched', denoised)
    denoised2 = cv2.fastNlMeansDenoisingColored(x1, None, 6, 10, 7, 21)
    # cv2.imshow('half noised', denoised2)
    # original = (255-original)
    # cv2.imshow('inverted', original)
    # HSV_applied = HSV_eq(original)
    # cv2.imshow('HSV_applied', HSV_applied)
    x2 = np.zeros(img.shape, img.dtype)
    x2 = cv2.addWeighted(denoised, 0.4, x1, 0.6, 0.0, x);
    # cv2.imshow('final_product', x2)
    # cv2.imwrite('final_product.jpg', x2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return x2


# def contrast_stretch(img):
#     xp = [0, 64, 128, 192, 255]
#     fp = [0, 16, 128, 240, 255]
#     x = np.arange(256)
#     table = np.interp(x, xp, fp).astype('uint8')
#     img = cv2.LUT(img, table)
#     return img

def inc(img):
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    # fora = cv2.createCLAHE(clipLimit=2., tileGridSize=(12,12))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    # l2 = l*2
    # b = fora.apply(b)
    lab = cv2.merge((l2, a, b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img2

def inc2(img,x =1.5):
    clahe = cv2.createCLAHE(clipLimit=x, tileGridSize=(8, 8))
    # fora = cv2.createCLAHE(clipLimit=2., tileGridSize=(12,12))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    # l2 = l*2
    # b = fora.apply(b)
    lab = cv2.merge((l2, a, b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img2


def Brightness(image, alpha, beta):
    new_image = np.zeros(image.shape, image.dtype)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)
    return new_image


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def HSV_eq(img):
    img_transf = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_transf[:, :, 2] = cv2.equalizeHist(img_transf[:, :, 2])
    img4 = cv2.cvtColor(img_transf, cv2.COLOR_HSV2BGR)
    return img4
