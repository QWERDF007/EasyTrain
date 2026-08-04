import torch

import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve, \
    average_precision_score
import cv2
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from functools import partial
import math
import warnings
from adeval import EvalAccumulatorCuda


def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x


def global_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        if stop_grad:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1).detach(),
                                            b[item].view(b[item].shape[0], -1)))
        else:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))
    loss = loss / len(a)
    return loss


def global_cosine_hm_percent(a, b, p=0.9, factor=0.):
    cos_loss = F.cosine_similarity

    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - F.cosine_similarity(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


def map_normalization(fs_list, ft_list, start=0.5, end=0.95):
    start_list = []
    end_list = []
    with torch.no_grad():
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            a_map = 1 - F.cosine_similarity(fs, ft)
            start_list.append(torch.quantile(a_map, q=start).item())
            end_list.append(torch.quantile(a_map, q=end).item())

    return [start_list, end_list]


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N


def evaluation_batch(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            # anomaly_map = anomaly_map - anomaly_map.mean(dim=[1, 2, 3]).view(-1, 1, 1, 1)

            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')

            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)

                # sp_score = sp_score + sp_score - anomaly_map.mean(dim=1)

            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        aupro_px = compute_pro(gt_list_px, pr_list_px)

        gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()

        auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        ap_px = average_precision_score(gt_list_px, pr_list_px)
        ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

        f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def evaluation_batch_noseg(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None,
                           cal_anomaly_maps_func=cal_anomaly_maps):
    model.eval()
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, label, img_path in dataloader:
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]

            anomaly_map, _ = cal_anomaly_maps_func(en, de, img.shape[-1])

            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)

            anomaly_map = gaussian_kernel(anomaly_map)

            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)

            # if cls_token_ano:
            #     en_cls_token, de_cls_token = output[2], output[3]
            #     sp_score = 1 - torch.cosine_similarity(en_cls_token, de_cls_token, dim=-1)

            pr_list_sp.append(sp_score)

        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
        f1_sp = f1_score_max(gt_list_sp, pr_list_sp)

    return [auroc_sp, ap_sp, f1_sp]


def evaluation_batch_multiview(model, dataloader, device, _class_=None, max_ratio_image=0, max_ratio_object=0,
                               resize_mask=None, multi_view_model=False, use_max_view=False, pixel_level_metrics=True,
                               adeval=False):
    model.eval()

    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []

    gt_list_obj = []  # object-level ground truth
    pr_list_obj = []  # object-level predictions

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for batch_data in dataloader:
            # Multi-view模式: img=[B,V,C,H,W], gt=[B,V,1,H,W], label=[B,V], img_path=[B]
            img, gt, label, img_path = batch_data
            img = img.to(device)

            B, V, C, H, W = img.shape

            # 将multi-view数据reshape为batch维度 [B*V, C, H, W]
            if multi_view_model:
                output = model(img)
                if isinstance(output, list) or isinstance(output, tuple):
                    en, de = output[0], output[1]

                    en = [e.view(B * V, -1, e.shape[3], e.shape[4]) for e in en]
                    de = [d.view(B * V, -1, d.shape[3], d.shape[4]) for d in de]
                    anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
                else:
                    anomaly_map = output

            else:
                img_reshaped = img.view(-1, C, H, W).to(device)
                output = model(img_reshaped)

                if isinstance(output, list) or isinstance(output, tuple):
                    en, de = output[0], output[1]
                    anomaly_map, _ = cal_anomaly_maps(en, de, img_reshaped.shape[-1])
                else:
                    anomaly_map = output

            gt_reshaped = gt.view(-1, gt.shape[2], H, W)  # [B*V, 1, H, W]

            # 调整大小
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt_reshaped = F.interpolate(gt_reshaped, size=resize_mask, mode='nearest')

            # 高斯平滑
            anomaly_map = gaussian_kernel(anomaly_map)
            gt_reshaped = gt_reshaped.bool()

            if gt_reshaped.shape[1] > 1:
                gt_reshaped = torch.max(gt_reshaped, dim=1, keepdim=True)[0]

            # 将结果重新reshape回multi-view格式 [B, V, 1, H, W]
            anomaly_map = anomaly_map.view(B, V, 1, anomaly_map.shape[-2], anomaly_map.shape[-1])
            gt_reshaped = gt_reshaped.view(B, V, 1, gt_reshaped.shape[-2], gt_reshaped.shape[-1])

            # 收集pixel-level和image-level数据（所有view展平）
            gt_list_px.append(gt_reshaped.view(-1, 1, gt_reshaped.shape[-2], gt_reshaped.shape[-1]).cpu())
            pr_list_px.append(anomaly_map.view(-1, 1, anomaly_map.shape[-2], anomaly_map.shape[-1]).cpu())

            # Image-level scores for all views
            image_level_scores = []
            image_level_toppx_scores = []
            for v in range(V):
                if max_ratio_image == 0:
                    sp_score = torch.max(anomaly_map[:, v].flatten(1), dim=1)[0]
                else:
                    anomaly_flat = anomaly_map[:, v].flatten(1)
                    sp_score_toppx = torch.sort(anomaly_flat, dim=1, descending=True)[0][:,
                                     :int(anomaly_flat.shape[1] * max_ratio_image)]
                    image_level_toppx_scores.append(sp_score_toppx)

                    sp_score = sp_score_toppx.mean(dim=1)
                image_level_scores.append(sp_score)
            all_view_scores = torch.stack(image_level_scores, dim=1)  # [B, V]

            # Object-level scores for all views
            if use_max_view:
                obj_scores = torch.max(all_view_scores, dim=1)[0]
            else:
                if max_ratio_object == 0:
                    obj_scores = torch.max(anomaly_map.flatten(1), dim=1)[0]
                else:
                    anomaly_flat = anomaly_map.flatten(1)
                    obj_scores = torch.sort(anomaly_flat, dim=1, descending=True)[0][:,
                                 :int(anomaly_flat.shape[1] * max_ratio_object)]
                    obj_scores = obj_scores.mean(dim=1)

            # 所有view的image-level scores
            gt_list_sp.append(label.flatten())  # 每个view都用相同的object label
            pr_list_sp.append(all_view_scores.flatten())

            gt_list_obj.append(label.max(dim=1)[0])  # object-level ground truth
            pr_list_obj.append(obj_scores)

    gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
    pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()
    gt_list_obj = torch.cat(gt_list_obj).flatten().cpu().numpy()
    pr_list_obj = torch.cat(pr_list_obj).flatten().cpu().numpy()

    if pixel_level_metrics:
        if adeval:
            gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cuda().byte()
            pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cuda()
            score_min, score_max = pr_list_px.min().item(), pr_list_px.max().item()
            accum = EvalAccumulatorCuda(score_min, score_max, score_min, score_max)
            accum.add_anomap_batch(pr_list_px, gt_list_px)
            metrics = accum.summary()
            auroc_px, ap_px, f1_px, aupro_px = metrics['p_auroc'], metrics['p_aupr'], \
                metrics['p_f1max'], metrics['p_aupro']
        else:
            gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
            pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
            aupro_px = compute_pro(gt_list_px, pr_list_px)
            gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()

            auroc_px = roc_auc_score(gt_list_px, pr_list_px)
            ap_px = average_precision_score(gt_list_px, pr_list_px)
            f1_px = f1_score_max(gt_list_px, pr_list_px)
    else:
        auroc_px, ap_px, f1_px, aupro_px = 0, 0, 0, 0

    auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
    ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
    f1_sp = f1_score_max(gt_list_sp, pr_list_sp)

    auroc_obj = roc_auc_score(gt_list_obj, pr_list_obj)
    ap_obj = average_precision_score(gt_list_obj, pr_list_obj)
    f1_obj = f1_score_max(gt_list_obj, pr_list_obj)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, auroc_obj, ap_obj, f1_obj]


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        try:
            df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        except:
            df = df._append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def evaluation_batch_rgbinfra(model, dataloader, device, class_=None, max_ratio=0, resize_mask=None):
    model.eval()

    # 像素级别的列表 - 分别存储RGB和红外的结果
    gt_list_px_rgb = []
    pr_list_px_rgb = []
    gt_list_px_infra = []
    pr_list_px_infra = []

    # 对象级别的列表
    gt_list_obj = []
    pr_list_obj = []

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for (rgb, infra), label, (rgb_mask, infra_mask), (rgb_path, infra_path) in dataloader:
            rgb = rgb.to(device)
            infra = infra.to(device)

            # label=1 if any modal is anomaly
            label = torch.stack(label)
            label = label.max(dim=0)[0]
            # 分别对RGB和红外图像进行推理
            rgb_output = model(rgb)
            infra_output = model(infra)

            # 提取编码器和解码器输出
            rgb_en, rgb_de = rgb_output[0], rgb_output[1]
            infra_en, infra_de = infra_output[0], infra_output[1]

            # 计算异常图
            rgb_anomaly_map, _ = cal_anomaly_maps(rgb_en, rgb_de, rgb.shape[-1])
            infra_anomaly_map, _ = cal_anomaly_maps(infra_en, infra_de, infra.shape[-1])

            # 如果需要调整mask大小
            if resize_mask is not None:
                rgb_anomaly_map = F.interpolate(rgb_anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                infra_anomaly_map = F.interpolate(infra_anomaly_map, size=resize_mask, mode='bilinear',
                                                  align_corners=False)
                rgb_mask = F.interpolate(rgb_mask, size=resize_mask, mode='nearest')
                infra_mask = F.interpolate(infra_mask, size=resize_mask, mode='nearest')

            # 应用高斯核
            rgb_anomaly_map = gaussian_kernel(rgb_anomaly_map)
            infra_anomaly_map = gaussian_kernel(infra_anomaly_map)

            # 处理ground truth
            rgb_gt = rgb_mask.bool()
            infra_gt = infra_mask.bool()

            if rgb_gt.shape[1] > 1:
                rgb_gt = torch.max(rgb_gt, dim=1, keepdim=True)[0]
            if infra_gt.shape[1] > 1:
                infra_gt = torch.max(infra_gt, dim=1, keepdim=True)[0]

            # 存储像素级别的结果
            gt_list_px_rgb.append(rgb_gt)
            pr_list_px_rgb.append(rgb_anomaly_map)
            gt_list_px_infra.append(infra_gt)
            pr_list_px_infra.append(infra_anomaly_map)

            # 计算图像级别的分数
            if max_ratio == 0:
                rgb_sp_score = torch.max(rgb_anomaly_map.flatten(1), dim=1)[0]
                infra_sp_score = torch.max(infra_anomaly_map.flatten(1), dim=1)[0]
            else:
                rgb_anomaly_flat = rgb_anomaly_map.flatten(1)
                infra_anomaly_flat = infra_anomaly_map.flatten(1)

                rgb_sp_score = torch.sort(rgb_anomaly_flat, dim=1, descending=True)[0][:,
                               :int(rgb_anomaly_flat.shape[1] * max_ratio)]
                rgb_sp_score = rgb_sp_score.mean(dim=1)

                infra_sp_score = torch.sort(infra_anomaly_flat, dim=1, descending=True)[0][:,
                                 :int(infra_anomaly_flat.shape[1] * max_ratio)]
                infra_sp_score = infra_sp_score.mean(dim=1)

            # 合并RGB和红外的图像级别分数作为对象级别分数
            obj_score = (rgb_sp_score + infra_sp_score)
            # obj_score = 1 - (1 - rgb_sp_score) * (1 - infra_sp_score)

            # 存储对象级别的结果
            gt_list_obj.append(label)
            pr_list_obj.append(obj_score)

    # 转换为numpy数组进行评估
    gt_list_px_rgb = torch.cat(gt_list_px_rgb, dim=0)[:, 0].cpu().numpy()
    pr_list_px_rgb = torch.cat(pr_list_px_rgb, dim=0)[:, 0].cpu().numpy()
    gt_list_px_infra = torch.cat(gt_list_px_infra, dim=0)[:, 0].cpu().numpy()
    pr_list_px_infra = torch.cat(pr_list_px_infra, dim=0)[:, 0].cpu().numpy()

    gt_list_obj = torch.cat(gt_list_obj).flatten().cpu().numpy()
    pr_list_obj = torch.cat(pr_list_obj).flatten().cpu().numpy()

    # 计算像素级别指标 - RGB
    aupro_px_rgb = compute_pro(gt_list_px_rgb, pr_list_px_rgb)
    gt_list_px_rgb, pr_list_px_rgb = gt_list_px_rgb.ravel(), pr_list_px_rgb.ravel()
    auroc_px_rgb = roc_auc_score(gt_list_px_rgb, pr_list_px_rgb)
    ap_px_rgb = average_precision_score(gt_list_px_rgb, pr_list_px_rgb)
    f1_px_rgb = f1_score_max(gt_list_px_rgb, pr_list_px_rgb)

    # 计算像素级别指标 - 红外
    aupro_px_infra = compute_pro(gt_list_px_infra, pr_list_px_infra)
    gt_list_px_infra, pr_list_px_infra = gt_list_px_infra.ravel(), pr_list_px_infra.ravel()
    auroc_px_infra = roc_auc_score(gt_list_px_infra, pr_list_px_infra)
    ap_px_infra = average_precision_score(gt_list_px_infra, pr_list_px_infra)
    f1_px_infra = f1_score_max(gt_list_px_infra, pr_list_px_infra)

    # 计算对象级别指标
    auroc_obj = roc_auc_score(gt_list_obj, pr_list_obj)
    ap_obj = average_precision_score(gt_list_obj, pr_list_obj)
    f1_obj = f1_score_max(gt_list_obj, pr_list_obj)

    return {
        'object_level': [auroc_obj, ap_obj, f1_obj],
        'pixel_rgb': [auroc_px_rgb, ap_px_rgb, f1_px_rgb, aupro_px_rgb],
        'pixel_infra': [auroc_px_infra, ap_px_infra, f1_px_infra, aupro_px_infra]
    }


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class FeatureJitter(torch.nn.Module):
    def __init__(self, scale=1., p=0.25) -> None:
        super(FeatureJitter, self).__init__()
        self.scale = scale
        self.p = p

    def add_jitter(self, feature):
        if self.scale > 0:
            B, C, H, W = feature.shape
            feature_norms = feature.norm(dim=1).unsqueeze(1) / C  # B*1*H*W
            jitter = torch.randn((B, C, H, W), device=feature.device)
            jitter = F.normalize(jitter, dim=1)
            jitter = jitter * feature_norms * self.scale
            mask = torch.rand((B, 1, H, W), device=feature.device) < self.p
            feature = feature + jitter * mask
        return feature

    def forward(self, x):
        if self.training:
            x = self.add_jitter(x)
        return x


def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup Cosine Scheduler for multiple parameter groups.
    For each parameter group:
    - First applies linear warmup for warmup_epochs
    - Then applies cosine annealing to reach final_lr = initial_lr * final_ratio
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, final_ratio=0.01, last_epoch=-1, verbose=False):
        """
        Args:
            optimizer: PyTorch optimizer with parameter groups
            warmup_epochs: Number of epochs for linear warmup
            total_epochs: Total number of training epochs
            final_ratio: Final learning rate ratio relative to initial learning rate
            last_epoch: The index of the last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.final_ratio = final_ratio
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        super(WarmupCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            return [initial_lr * (self.last_epoch / self.warmup_epochs) for initial_lr in self.initial_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            coeff = 0.5 * (1.0 + math.cos(math.pi * progress))

            return [initial_lr * (self.final_ratio + (1.0 - self.final_ratio) * coeff)
                    for initial_lr in self.initial_lrs]


def gaussian(window_size: int, sigma: float = 1.5):
    gauss = torch.Tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # [w, 1]
    window = (_1D_window @ _1D_window.T)[None, None]  # [1, 1, w, w]
    window = window.expand(channel, 1, window_size, window_size)
    return window
