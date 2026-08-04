import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
# from prepare_data.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
# import trimesh
import open3d as o3d
import csv
import re
from scipy.spatial import KDTree
import torch
import cv2
import matplotlib.pyplot as plt
from dataset import get_data_transforms


def mulsen_classes():
    return [
        "capsule",
        "cotton",
        "cube",
        "spring_pad",
        "screw",
        "screen",
        "piggy",
        "nut",
        "flat_pad",
        'plastic_cylinder',
        "zipper",
        "button_cell",
        "toothbrush",
        "solar_panel",
        "light",
    ]


def stl_to_depth_map(stl_path, image_size=(224, 224), normalize=True):
    """
    将STL文件转换为2D深度图

    Args:
        stl_path: STL文件路径
        image_size: 输出深度图尺寸 (height, width)
        normalize: 是否标准化深度值到[0,1]

    Returns:
        depth_map: numpy array形状为(H, W)的深度图
    """
    # 读取STL文件
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh = mesh.remove_duplicated_vertices()
    vertices = np.asarray(mesh.vertices)

    if len(vertices) == 0:
        return np.zeros(image_size)

    # 获取点云的边界
    x_min, y_min, z_min = vertices.min(axis=0)
    x_max, y_max, z_max = vertices.max(axis=0)

    # 投影到XY平面，使用Z作为深度
    # 将XY坐标映射到图像坐标系
    height, width = image_size

    # 避免除零错误
    x_range = max(x_max - x_min, 1e-8)
    y_range = max(y_max - y_min, 1e-8)

    # 将3D坐标映射到2D图像坐标
    u = ((vertices[:, 0] - x_min) / x_range * (width - 1)).astype(np.int32)
    v = ((vertices[:, 1] - y_min) / y_range * (height - 1)).astype(np.int32)

    # 确保坐标在有效范围内
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    # 创建深度图
    depth_map = np.full(image_size, -np.inf, dtype=np.float32)

    # 对于每个像素位置，保留最大的Z值（最前面的点）
    for i in range(len(vertices)):
        if depth_map[v[i], u[i]] < vertices[i, 2]:
            depth_map[v[i], u[i]] = vertices[i, 2]

    # 处理没有点投影到的像素（插值或设为最小值）
    mask = depth_map == -np.inf
    if mask.any():
        # 用最小深度值填充空洞
        valid_depths = depth_map[~mask]
        if len(valid_depths) > 0:
            depth_map[mask] = valid_depths.min()
        else:
            depth_map[mask] = 0

    # 标准化深度值到[0,1]
    if normalize and depth_map.max() > depth_map.min():
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    elif normalize:
        depth_map = np.zeros_like(depth_map)

    return depth_map


def create_depth_mask_from_3d_mask(stl_path, anomaly_points, image_size=(224, 224), tolerance=0.01):
    """
    根据3D异常点创建2D深度图掩码

    Args:
        stl_path: 原始STL文件路径
        anomaly_points: 3D异常点坐标 (N, 3)
        image_size: 输出掩码尺寸
        tolerance: 异常点匹配容忍度

    Returns:
        mask: 2D掩码 (H, W)，1表示异常区域
    """
    # 读取STL文件
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh = mesh.remove_duplicated_vertices()
    vertices = np.asarray(mesh.vertices)

    if len(vertices) == 0 or len(anomaly_points) == 0:
        return np.zeros(image_size)

    # 获取点云的边界
    x_min, y_min, z_min = vertices.min(axis=0)
    x_max, y_max, z_max = vertices.max(axis=0)

    height, width = image_size
    x_range = max(x_max - x_min, 1e-8)
    y_range = max(y_max - y_min, 1e-8)

    # 使用KDTree找到最近的异常点
    tree = KDTree(vertices)
    mask = np.zeros(image_size, dtype=np.float32)

    for anomaly_point in anomaly_points:
        # 找到最近的顶点
        distances, indices = tree.query(anomaly_point, k=10)  # 查找最近的10个点

        for dist, idx in zip(distances, indices):
            if dist < tolerance:
                vertex = vertices[idx]
                # 将3D点投影到2D
                u = int((vertex[0] - x_min) / x_range * (width - 1))
                v = int((vertex[1] - y_min) / y_range * (height - 1))

                # 确保坐标在有效范围内
                u = np.clip(u, 0, width - 1)
                v = np.clip(v, 0, height - 1)

                # 在掩码上标记异常区域（可以扩展到周围像素）
                mask[max(0, v - 1):min(height, v + 2), max(0, u - 1):min(width, u + 2)] = 1

    return mask


class BaseAnomalyDetectionDataset(Dataset):
    def __init__(self, split, class_name, transform, gt_transform, depth=False, dataset_path=''):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.img_path = os.path.join(dataset_path, self.cls)
        self.rgb_transform = transform
        self.depth_transform = transform
        self.gt_transform = gt_transform
        self.depth = depth

    def sort(self, file_paths):
        paths_with_numbers = []
        pattern = re.compile(r'(\d+)\.(png|stl)$')
        for path in file_paths:
            match = pattern.search(path)
            if match:
                number = int(match.group(1))
                paths_with_numbers.append((path, number))
        paths_with_numbers.sort(key=lambda x: x[1])
        return [p[0] for p in paths_with_numbers]


class TrainDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, transform, gt_transform, depth=False, dataset_path=''):
        super().__init__(split="train", class_name=class_name, dataset_path=dataset_path,
                         transform=transform, gt_transform=gt_transform, depth=depth)
        self.img_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'RGB', 'train') + "/*.png")
        infra_paths = glob.glob(os.path.join(self.img_path, 'Infrared', 'train') + "/*.png")
        pc_paths = glob.glob(os.path.join(self.img_path, 'Pointcloud', 'train') + "/*.stl")

        rgb_paths = self.sort(rgb_paths)
        infra_paths = self.sort(infra_paths)
        pc_paths = self.sort(pc_paths)
        sample_paths = list(zip(rgb_paths, infra_paths, pc_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        infra_path = img_path[1]
        pc_path = img_path[2]

        # 读取RGB和红外图像
        img = Image.open(rgb_path).convert('RGB')
        infra = Image.open(infra_path).convert('RGB')
        img = self.rgb_transform(img)
        infra = self.rgb_transform(infra)

        if self.depth:
            depth_map = stl_to_depth_map(pc_path, image_size=(img.shape[1], img.shape[2]))
            depth_map = self.depth_transform(depth_map)

            return (img, infra, depth_map), label

        return (img, infra), label


class TestDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, transform, gt_transform, depth=False, dataset_path=''):
        super().__init__(split="test", class_name=class_name, dataset_path=dataset_path,
                         transform=transform, gt_transform=gt_transform, depth=depth)
        self.img_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(os.path.join(self.img_path, 'RGB', 'test'))

        for defect_type in defect_types:
            label_rgb = []
            label_infra = []
            label_pc = []

            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, 'RGB', 'test', defect_type) + "/*.png")
                infra_paths = glob.glob(os.path.join(self.img_path, 'Infrared', 'test', defect_type) + "/*.png")
                pc_paths = glob.glob(os.path.join(self.img_path, 'Pointcloud', 'test', defect_type) + "/*.stl")

                rgb_paths = self.sort(rgb_paths)
                infra_paths = self.sort(infra_paths)
                pc_paths = self.sort(pc_paths)

                sample_paths = list(zip(rgb_paths, infra_paths, pc_paths))
                img_tot_paths.extend(sample_paths)

                label_rgb.extend([0] * len(sample_paths))
                label_infra.extend([0] * len(sample_paths))
                label_pc.extend([0] * len(sample_paths))
                label = list(zip(label_rgb, label_infra, label_pc))
                tot_labels.extend(label)
            else:
                with open(os.path.join(self.img_path, 'RGB', 'GT', defect_type, 'data.csv'), 'r') as file:
                    csvreader = csv.reader(file)
                    header = next(csvreader)
                    for row in csvreader:
                        object, label1, label2, label3 = row
                        label_rgb.extend([int(label1)])
                        label_infra.extend([int(label2)])
                        label_pc.extend([int(label3)])
                label = list(zip(label_rgb, label_infra, label_pc))
                tot_labels.extend(label)

                rgb_paths = glob.glob(os.path.join(self.img_path, 'RGB', 'test', defect_type) + "/*.png")
                infra_paths = glob.glob(os.path.join(self.img_path, 'Infrared', 'test', defect_type) + "/*.png")
                pc_paths = glob.glob(os.path.join(self.img_path, 'Pointcloud', 'test', defect_type) + "/*.stl")

                rgb_paths = self.sort(rgb_paths)
                infra_paths = self.sort(infra_paths)
                pc_paths = self.sort(pc_paths)
                sample_paths = list(zip(rgb_paths, infra_paths, pc_paths))

                img_tot_paths.extend(sample_paths)

        assert len(img_tot_paths) == len(tot_labels), "Something wrong with test and ground truth pair!"
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        infra_path = img_path[1]
        pc_path = img_path[2]

        # 读取RGB和红外图像
        img = Image.open(rgb_path).convert('RGB')
        infra = Image.open(infra_path).convert('RGB')

        img = self.rgb_transform(img)
        infra = self.rgb_transform(infra)
        # 处理掩码
        # RGB掩码
        if label[0] == 0:
            img_mask = torch.zeros([1, img.shape[1], img.shape[2]])
        else:
            img_mask_path = img_path[0].replace("test", "GT")
            img_mask = Image.open(img_mask_path).convert('L')
            img_mask = self.gt_transform(img_mask)
            img_mask = torch.where(img_mask > 0.5, 1., .0)

        # 红外掩码
        if label[1] == 0:
            infra_mask = torch.zeros([1, infra.shape[1], infra.shape[2]])
        else:
            infra_mask_path = img_path[1].replace("test", "GT")
            infra_mask = Image.open(infra_mask_path).convert('L')
            infra_mask = self.gt_transform(infra_mask)
            infra_mask = torch.where(infra_mask > 0.5, 1., .0)

        if self.depth:
            # 将STL转换为深度图
            RGB_SIZE = img.shape[1]
            depth_map = stl_to_depth_map(pc_path, image_size=(RGB_SIZE, RGB_SIZE))

            depth_map = self.depth_transform(depth_map)
            # 深度图掩码
            if label[2] == 0:
                depth_mask = torch.zeros([1, RGB_SIZE, RGB_SIZE])
            else:
                # 读取异常点坐标
                pointcloud_mask_path = img_path[2].replace("test", "GT").replace(".stl", ".txt")
                try:
                    pcd = np.genfromtxt(pointcloud_mask_path, delimiter=",")
                    if pcd.ndim == 1:  # 只有一个点的情况
                        pcd = pcd.reshape(1, -1)
                    anomaly_points = pcd[:, :3]

                    # 创建2D深度掩码
                    depth_mask_2d = create_depth_mask_from_3d_mask(
                        pc_path, anomaly_points,
                        image_size=(RGB_SIZE, RGB_SIZE)
                    )
                    depth_mask = torch.from_numpy(depth_mask_2d).unsqueeze(0).float()
                except:
                    # 如果读取异常点失败，创建空掩码
                    depth_mask = torch.zeros([1, RGB_SIZE, RGB_SIZE])

            return (img, infra, depth_map), label, (img_mask, infra_mask, depth_mask), (rgb_path, infra_path, pc_path)

        return (img, infra), label, (img_mask, infra_mask), (rgb_path, infra_path)



# 测试代码
if __name__ == '__main__':
    data_transform, gt_transform = get_data_transforms(448, 392)

    for cls in mulsen_classes():
        print(f"Testing class: {cls}")
        Test_loader = TestDataset(class_name=cls, dataset_path='../MulSen_AD',
                                  transform=data_transform, gt_transform=gt_transform, depth=False)

        for (img, infra), label, (img_mask, infra_mask), (rgb_path, infra_path) in Test_loader:
            print("Image Shape:", img.shape)
            print("Infrared Shape:", infra.shape)
            print("Label:", label)
            print("Image Mask Shape:", img_mask.shape)
            print("Infrared Mask Shape:", infra_mask.shape)
            break

        #
        # for (img, infra, depth_map), label, (img_mask, infra_mask, depth_mask), (
        #         rgb_path, infra_path, pc_path) in Test_loader:
        #     print("Image Shape:", img.shape)
        #     print("Infrared Shape:", infra.shape)
        #     print("Depth Map Shape:", depth_map.shape)
        #     print("Label:", label)
        #     print("Image Mask Shape:", img_mask.shape)
        #     print("Infrared Mask Shape:", infra_mask.shape)
        #     print("Depth Mask Shape:", depth_mask.shape)
        #     break
        #
