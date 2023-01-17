import numpy as np
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import mean_squared_error as sk_mse

def mse(image_A, image_B):
    # return np.mean((image_A - image_B) ** 2)
    return sk_mse(image_A, image_B)

def ssim(image_A, image_B):
    return sk_ssim(image_A, image_B, channel_axis=-1, data_range=255)

def psnr(image_A, image_B):
    return sk_psnr(image_A, image_B, data_range=255)


if __name__ == "__main__":
    fake_image_A, fake_image_B = np.random.rand(256, 256, 3) * 255, np.random.rand(256, 256, 3) * 255
    print(f'mse: {mse(fake_image_A, fake_image_B)} ssim: {ssim(fake_image_A, fake_image_B)} psnr: {psnr(fake_image_A, fake_image_B)}')
