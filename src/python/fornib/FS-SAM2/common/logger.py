r""" Logging during training/testing """
import datetime
import logging
import os

import torch
from .utils import is_main_process, save_on_master, reduce_metric


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        self.class_ids_interest = dataset.class_ids
        self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
        self.nclass = dataset.nclass

        self.intersection_buf = torch.zeros([2, self.nclass]).float().cuda()  # x=0: background, x=1: foreground
        self.union_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.pred_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.gt_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []

    def update(self, inter_b, union_b, pred_b, gt_b, class_id, loss=None):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())
        self.pred_buf.index_add_(1, class_id, pred_b.float())
        self.gt_buf.index_add_(1, class_id, gt_b.float())
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_iou(self):
        iou = self.intersection_buf.float() / torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou
    
    def compute_f1(self):
        precision = self.intersection_buf.float() / torch.max(torch.stack([self.pred_buf, self.ones]), dim=0)[0]
        recall = self.intersection_buf.float() / torch.max(torch.stack([self.gt_buf, self.ones]), dim=0)[0]
        f1 = 2 * (precision * recall) / torch.max(torch.stack([precision + recall, self.ones]), dim=0)[0]
        f1 = f1.index_select(1, self.class_ids_interest)
        mf1 = f1[1].mean() * 100

        return mf1

    def write_result(self, split, epoch):
        self.intersection_buf, self.union_buf = self.reduce_metrics([self.intersection_buf, self.union_buf], False)
        iou, fb_iou = self.compute_iou()
        mf1 = self.compute_f1()

        # loss_buf = torch.stack(self.loss_buf)
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch if epoch != -1 else ''
        if epoch != -1:
            loss_buf = torch.stack(self.loss_buf)
            loss_buf = self.reduce_metrics([loss_buf])[0]
            msg += 'Avg L: %6.5f  ' % loss_buf.mean()
        ###msg += 'mIoU: %5.2f   ' % iou
        iou_classwise = self.intersection_buf.float() / torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou_classwise = iou_classwise.index_select(1, self.class_ids_interest)[1]
        iou_classwise_str = ",".join(f'{x*100:5.2f}' for x in iou_classwise)
        msg += f'mIoU: {iou:5.2f} ({iou_classwise_str})  '
        ###
        msg += 'FB-IoU: %5.2f   ' % fb_iou
        ###
        precision = self.intersection_buf.float() / torch.max(torch.stack([self.pred_buf, self.ones]), dim=0)[0]
        recall = self.intersection_buf.float() / torch.max(torch.stack([self.gt_buf, self.ones]), dim=0)[0]
        f1_classwise = 2 * (precision * recall) / torch.max(torch.stack([precision + recall, self.ones]), dim=0)[0]
        f1_classwise = f1_classwise.index_select(1, self.class_ids_interest)[1]
        f1_classwise_str = ",".join(f'{x*100:5.2f}' for x in f1_classwise)
        msg += f'mF1: {mf1:5.2f} ({f1_classwise_str})  '
        ###

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20, dt=0):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            iou, fb_iou = self.compute_iou()
            mf1 = self.compute_f1()
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            ###msg += 'mIoU: %5.2f   ' % iou
            iou_classwise = self.intersection_buf.float() / torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
            iou_classwise = iou_classwise.index_select(1, self.class_ids_interest)[1]
            iou_classwise_str = ",".join(f'{x*100:5.2f}' for x in iou_classwise)
            msg += f'mIoU: {iou:5.2f} ({iou_classwise_str})  '
            ###
            msg += 'FB-IoU: %5.2f  |  ' % fb_iou
            ###
            precision = self.intersection_buf.float() / torch.max(torch.stack([self.pred_buf, self.ones]), dim=0)[0]
            recall = self.intersection_buf.float() / torch.max(torch.stack([self.gt_buf, self.ones]), dim=0)[0]
            f1_classwise = 2 * (precision * recall) / torch.max(torch.stack([precision + recall, self.ones]), dim=0)[0]
            f1_classwise = f1_classwise.index_select(1, self.class_ids_interest)[1]
            f1_classwise_str = ",".join(f'{x*100:5.2f}' for x in f1_classwise)
            msg += f'mF1: {mf1:5.2f} ({f1_classwise_str})  '
            ###
            msg += 'time: %5.3f' % dt
            Logger.info(msg)
    
    def reduce_metrics(self, metrics, average=True):
        reduced_metrics = []
        for m in metrics:
            reduce_metric(m, average)
            reduced_metrics.append(m)
        return reduced_metrics


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().strftime(r'%Y-%m-%d_%H:%M:%S')
        logpath = args.logpath if training else args.logpath + '-TEST'
        if logpath == '':
            logpath = logtime

        cls.logpath = os.path.join('logs', logpath + '.log')
        cls.benchmark = args.benchmark
        os.makedirs(cls.logpath, exist_ok=True)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        from torch.utils.tensorboard import SummaryWriter
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:=========== Few-shot Seg. with FS-SAM2 ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':==================================================\n')
        logging.info(f'Datetime: {logtime}')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        state_dict = model.state_dict()
        checkpoint = {
            'state_dict': state_dict,
            #'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_miou': val_miou,
        }
        torch.save(checkpoint, os.path.join(cls.logpath, 'best_model.pt'))
        cls.info(f'Epoch {epoch} | Model saved at best_model.pt w/ val. mIoU: {val_miou:5.2f}.\n')

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for param in model.parameters():  #raw_model.state_dict().keys()
            n_param = param.numel()  #raw_model.state_dict()[k].view(-1).size(0)
            if param.requires_grad:
                learner_param += n_param
            else:
                backbone_param += n_param
        Logger.info(f'Backbone # param.: {backbone_param:,}')
        Logger.info(f'Learnable # param.: {learner_param:,}')
        Logger.info(f'Total # param.: {backbone_param + learner_param:,}')

