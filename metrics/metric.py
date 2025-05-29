import os
import numpy as np
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
import threading

def compute_mae(imageA, imageB):
    return np.sum(np.abs(imageA - imageB)) / float(imageA.shape[0] * imageA.shape[1])

def compute_metrics(filenames, dir1, dir2, results, lock):
    local_psnr = 0
    local_ssim = 0
    local_mae = 0

    for filename in filenames:
        original_path = os.path.join(dir1, filename)
        generated_path = os.path.join(dir2, filename)

        original = img_as_float(io.imread(original_path))
        generated = img_as_float(io.imread(generated_path))

        original_gray = rgb2gray(original)
        generated_gray = rgb2gray(generated)

        local_psnr += psnr(original, generated, data_range=original.max() - original.min())
        local_ssim += ssim(original_gray, generated_gray, data_range=generated_gray.max() - generated_gray.min())
        local_mae += compute_mae(original, generated)

    with lock:
        results.append((local_psnr, local_ssim, local_mae))


def test_dir(dir2):
    dir1 = '/data1/fyw/datasets/celeba/celeba_test_images'

    filenames = os.listdir(dir1)
    num_threads = 8
    chunk_size = len(filenames) // num_threads

    threads = []
    results = []
    lock = threading.Lock()

    for i in range(num_threads):
        start_index = i * chunk_size
        end_index = start_index + chunk_size if i != num_threads - 1 else len(filenames)
        thread_files = filenames[start_index:end_index]
        t = threading.Thread(target=compute_metrics, args=(thread_files, dir1, dir2, results, lock))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    total_psnr = sum([res[0] for res in results])
    total_ssim = sum([res[1] for res in results])
    total_mae = sum([res[2] for res in results])

    avg_psnr = total_psnr / len(filenames)
    avg_ssim = total_ssim / len(filenames)
    avg_mae = total_mae / len(filenames)
    
    print(dir2)
    print(f'Average PSNR: {avg_psnr}')
    print(f'Average SSIM: {avg_ssim}')
    print(f'Average MAE: {avg_mae}')
  
  
def main():
    test_dirs = [
      '/data1/fyw/OASG-NET/fin/CelebA_Pconv/results/landmark_inpaint/result',
      '/data1/fyw/OASG-NET/fin/CelebA_Loss_Compare/Gt2/results/landmark_inpaint/result',
      '/data1/fyw/OASG-NET/fin/CelebA_Loss_Compare/L1_0.5/results/landmark_inpaint/result',
      '/data1/fyw/OASG-NET/fin/CelebA_Loss_Compare/L1_2.0/results/landmark_inpaint/result',
      '/data1/fyw/OASG-NET/fin/CelebA_Loss_Compare/TV_0.5/results/landmark_inpaint/result',
      '/data1/fyw/OASG-NET/fin/CelebA_Loss_Compare/TV_0.02/results/landmark_inpaint/result',
      '/data1/fyw/OASG-NET/fin/CelebA_Loss_Compare/St_500/results/landmark_inpaint/result',
      '/data1/fyw/OASG-NET/fin/CelebA_Loss_Compare/St_125/results/landmark_inpaint/result',
      '/data1/fyw/OASG-NET/fin/CelebA_Loss_Compare/Adv_0.1/results/landmark_inpaint/result',
      '/data1/fyw/OASG-NET/fin/CelebA_Loss_Compare/Adv_0.001/results/landmark_inpaint/result'
    ]
    for dir_name in test_dirs:
      test_dir(dir_name)
    

if __name__ == '__main__':
    main()
