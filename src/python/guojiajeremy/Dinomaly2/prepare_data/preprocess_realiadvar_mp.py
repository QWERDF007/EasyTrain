import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


def process_single_image(img_path, src_path, dst_path, target_size):
    """
    处理单个图像

    Args:
        img_path: 图像路径
        src_path: 源数据集路径
        dst_path: 目标数据集路径
        target_size: 目标尺寸 (width, height)

    Returns:
        bool: 处理是否成功
    """
    try:
        # 计算相对路径
        rel_path = img_path.relative_to(src_path)
        output_path = dst_path / rel_path

        # 创建输出目录
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 读取图像
        img = Image.open(img_path)

        # 判断是否为mask文件
        is_mask = '_mask' in img_path.name

        if is_mask:
            # 处理mask：转为灰度图，resize，然后二值化
            if img.mode != 'L':
                img = img.convert('L')

            # 使用bicubic插值resize
            img_resized = img.resize(target_size, Image.BICUBIC)

            # 二值化：阈值设为127
            img_array = np.array(img_resized)
            img_binary = (img_array > 127).astype(np.uint8) * 255
            img_final = Image.fromarray(img_binary)

        else:
            # 处理普通图像：使用bicubic插值resize
            img_resized = img.resize(target_size, Image.BICUBIC)
            img_final = img_resized

        # 保存图像
        img_final.save(output_path)

        return True

    except Exception as e:
        print(f"\n处理 {img_path} 时出错: {str(e)}")
        return False


def process_dataset(src_dir, dst_dir, target_size=(512, 512), num_workers=None):
    """
    使用多线程处理数据集，将所有图像resize到指定尺寸

    Args:
        src_dir: 源数据集路径
        dst_dir: 目标数据集路径
        target_size: 目标尺寸 (width, height)
        num_workers: 线程数，默认为CPU核心数的2倍
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    # 设置线程数
    if num_workers is None:
        num_workers = multiprocessing.cpu_count() * 2

    print(f"使用 {num_workers} 个线程进行处理")

    # 获取所有png文件
    all_images = list(src_path.rglob('*.png'))
    print(f"找到 {len(all_images)} 个图像文件")

    # 使用线程池处理图像
    success_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_single_image, img_path, src_path, dst_path, target_size): img_path
            for img_path in all_images
        }

        # 使用tqdm显示进度
        with tqdm(total=len(all_images), desc="处理图像") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                else:
                    failed_count += 1
                pbar.update(1)

    print(f"\n处理完成！")
    print(f"成功: {success_count} 个")
    print(f"失败: {failed_count} 个")
    print(f"数据已保存到: {dst_dir}")


def main():
    # 设置路径
    src_dir = '/data1/gj/Real-IAD_Variety/realiadvariety_raw'
    dst_dir = '/data1/gj/Real-IAD_Variety/realiadvariety_1024'

    # 检查源目录是否存在
    if not os.path.exists(src_dir):
        print(f"错误：源目录 {src_dir} 不存在！")
        return

    # 处理数据集
    # 可以手动指定线程数，例如 num_workers=16
    # 不指定则默认使用 CPU核心数 * 2
    process_dataset(src_dir, dst_dir, target_size=(1024, 1024), num_workers=8)


if __name__ == '__main__':
    main()
