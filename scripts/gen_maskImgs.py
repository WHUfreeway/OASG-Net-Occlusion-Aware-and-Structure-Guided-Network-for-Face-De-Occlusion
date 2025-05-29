import os
import cv2
import insightface
from insightface.app import MaskRenderer

torch.cuda.set_device(0)

def extract_mask(original, rendered):
    # 计算原始图像和渲染后的图像之间的差异
    diff = cv2.absdiff(original, rendered)
    
    # 将差异转换为灰度
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 二值化图像以提取口罩
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    
    return mask

if __name__ == "__main__":
    # input_dir = "/data1/fyw/datasets/celeba/celeba_test_images/"
    # output_mask_dir = "/data1/fyw/datasets/celeba/celeba_test_masks/"
    # output_mask_only_dir = "/data1/fyw/datasets/celeba/celeba_test_mask_images/"
    #
    # # 确保输出目录存在
    # if not os.path.exists(output_mask_dir):
    #     os.makedirs(output_mask_dir)
    # if not os.path.exists(output_mask_only_dir):
    #     os.makedirs(output_mask_only_dir)
    #
    # tool = MaskRenderer()
    # tool.prepare(ctx_id=0, det_size=(128,128))
    #
    # for filename in os.listdir(input_dir):
    #     if filename.endswith(".png"):
    #         filepath = os.path.join(input_dir, filename)
    #         image = cv2.imread(filepath)
    #         mask_image  = "mask_blue"
    #         params = tool.build_params(image)
    #         # 检查params是否为None
    #         if params is None:
    #           print(f"Warning: Unable to build params for image {filename}. Skipping...")
    #           continue
    #         mask_out = tool.render_mask(image, mask_image, params)
    #
    #         # 保存mask_out
    #         cv2.imwrite(os.path.join(output_mask_dir, filename), mask_out)
    #
    #         # 提取并保存mask_only
    #         mask = extract_mask(image, mask_out)
    #         cv2.imwrite(os.path.join(output_mask_only_dir, filename), mask)
    #
    # input_dir = "/data1/fyw/datasets/celeba/celeba_test_images/"
    # output_mask_dir = "/data1/fyw/datasets/celeba/celeba_test_masks/"
    # output_mask_only_dir = "/data1/fyw/datasets/celeba/celeba_test_mask_images/"
    #
    # # 确保输出目录存在
    # if not os.path.exists(output_mask_dir):
    #     os.makedirs(output_mask_dir)
    # if not os.path.exists(output_mask_only_dir):
    #     os.makedirs(output_mask_only_dir)
    #
    # tool = MaskRenderer()
    # tool.prepare(ctx_id=0, det_size=(128,128))
    #
    # for filename in os.listdir(input_dir):
    #     if filename.endswith(".png"):
    #         filepath = os.path.join(input_dir, filename)
    #         image = cv2.imread(filepath)
    #         mask_image  = "mask_blue"
    #         params = tool.build_params(image)
    #         # 检查params是否为None
    #         if params is None:
    #           print(f"Warning: Unable to build params for image {filename}. Skipping...")
    #           continue
    #         mask_out = tool.render_mask(image, mask_image, params)
    #
    #         # 保存mask_out
    #         cv2.imwrite(os.path.join(output_mask_dir, filename), mask_out)
    #
    #         # 提取并保存mask_only
    #         mask = extract_mask(image, mask_out)
    #         cv2.imwrite(os.path.join(output_mask_only_dir, filename), mask)
    #
    #
    # input_dir = "/data1/fyw/datasets/celeba/celeba_train_images/"
    # output_mask_dir = "/data1/fyw/datasets/celeba/celeba_train_masks/"
    # output_mask_only_dir = "/data1/fyw/datasets/celeba/celeba_train_mask_images/"
    #
    # # 确保输出目录存在
    # if not os.path.exists(output_mask_dir):
    #     os.makedirs(output_mask_dir)
    # if not os.path.exists(output_mask_only_dir):
    #     os.makedirs(output_mask_only_dir)
    #
    # tool = MaskRenderer()
    # tool.prepare(ctx_id=0, det_size=(128,128))
    #
    # for filename in os.listdir(input_dir):
    #     if filename.endswith(".png"):
    #         filepath = os.path.join(input_dir, filename)
    #         image = cv2.imread(filepath)
    #         mask_image  = "mask_blue"
    #         params = tool.build_params(image)
    #         # 检查params是否为None
    #         if params is None:
    #           print(f"Warning: Unable to build params for image {filename}. Skipping...")
    #           continue
    #         mask_out = tool.render_mask(image, mask_image, params)
    #
    #         # 保存mask_out
    #         cv2.imwrite(os.path.join(output_mask_dir, filename), mask_out)
    #
    #         # 提取并保存mask_only
    #         mask = extract_mask(image, mask_out)
    #         cv2.imwrite(os.path.join(output_mask_only_dir, filename), mask)
    #
    #
    # input_dir = "/data1/fyw/datasets/celeba/celeban_val_images/"
    # output_mask_dir = "/data1/fyw/datasets/celeba/celeba_val_masks/"
    # output_mask_only_dir = "/data1/fyw/datasets/celeba/celeba_val_mask_images/"
    #
    # # 确保输出目录存在
    # if not os.path.exists(output_mask_dir):
    #     os.makedirs(output_mask_dir)
    # if not os.path.exists(output_mask_only_dir):
    #     os.makedirs(output_mask_only_dir)
    #
    # tool = MaskRenderer()
    # tool.prepare(ctx_id=0, det_size=(128,128))
    #
    # for filename in os.listdir(input_dir):
    #     if filename.endswith(".png"):
    #         filepath = os.path.join(input_dir, filename)
    #         image = cv2.imread(filepath)
    #         mask_image  = "mask_blue"
    #         params = tool.build_params(image)
    #         # 检查params是否为None
    #         if params is None:
    #           print(f"Warning: Unable to build params for image {filename}. Skipping...")
    #           continue
    #         mask_out = tool.render_mask(image, mask_image, params)
    #
    #         # 保存mask_out
    #         cv2.imwrite(os.path.join(output_mask_dir, filename), mask_out)
    #
    #         # 提取并保存mask_only
    #         mask = extract_mask(image, mask_out)
    #         cv2.imwrite(os.path.join(output_mask_only_dir, filename), mask)
