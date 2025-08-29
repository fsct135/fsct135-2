import os

from lifelines.utils import concordance_index as ci
import numpy as np
import torch
import math
import shutil

from sklearn.model_selection import KFold
from torch import nn
from lifelines.statistics import logrank_test


def cal_pval(time, pred):
    event = np.zeros_like(time)
    event[time > 0] = 1
    pred = np.squeeze(pred)  # 预测得分
    pred_median = np.median(pred)
    risk_group = np.zeros_like(pred)
    risk_group[pred > pred_median] = 1

    group_lowrisk_time = time[risk_group==0].copy()
    group_highrisk_time = time[risk_group==1].copy()
    group_lowrisk_event = event[risk_group==0].copy()
    group_highrisk_event = event[risk_group==1].copy()

    results = logrank_test(group_lowrisk_time, group_highrisk_time, event_observed_A=group_lowrisk_event , event_observed_B=group_highrisk_event)
    # results.print_summary()
    return results.p_value

def concordance_index(y_true, y_pred):
    """
    Compute the concordance-index value.

    Parameters
    ----------
    y_true : np.array
        Observed time. Negtive values are considered right censored.
    y_pred : np.array
        Predicted value.

    Returns
    -------
    float
        Concordance index.
    """

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)  # 预测得分
    t = np.abs(y_true)  # time
    e = (y_true > 0).astype(np.int32)  # event
    ci_value = ci(t, y_pred, e)
    return ci_value

def PartialLogLikelihood(logits, fail_indicator, ties):
    '''
    fail_indicator: 1 if the sample fails, 0 if the sample is censored.
    logits: raw output from model
    ties: 'noties' or 'efron' or 'breslow'
    '''
    logL = 0
    # pre-calculate cumsum
    cumsum_y_pred = torch.cumsum(logits, 0)
    hazard_ratio = torch.exp(logits)
    cumsum_hazard_ratio = torch.cumsum(hazard_ratio, 0)
    if ties == 'noties':
        log_risk = torch.log(cumsum_hazard_ratio)
        likelihood = logits - log_risk
        # dimension for E: np.array -> [None, 1]
        uncensored_likelihood = likelihood * fail_indicator
        logL = -torch.sum(uncensored_likelihood)
    else:
        raise NotImplementedError()
    # negative average log-likelihood
    observations = torch.sum(fail_indicator, 0)
    return 1.0*logL / observations

class DeepSurvCox_LossFunc(torch.nn.Module):
    def __init__(self):
        super(DeepSurvCox_LossFunc, self).__init__()


    def forward(self, y_predict, t):
        y_pred_list = y_predict.view(-1)
        y_pred_exp = torch.exp(y_pred_list)  # 返回e的x次方
        t_list = t.view(-1)
        t_E = torch.gt(t_list, 0)# 事件t
        cumsum_hazard_ratio = torch.cumsum(y_pred_exp, 0)

        log_risk = torch.log(cumsum_hazard_ratio)
        likelihood = y_pred_list - log_risk
        # dimension for E: np.array -> [None, 1]
        uncensored_likelihood = likelihood * t_E
        logL = -torch.sum(uncensored_likelihood)
        observations = torch.sum(t_E, 0)
        return 1.0 * logL / observations

class DeepCox_LossFunc(torch.nn.Module):
    def __init__(self):
        super(DeepCox_LossFunc, self).__init__()


    def forward(self, y_predict, t):



        y_pred_list = y_predict.view(-1)
        y_pred_exp = torch.exp(y_pred_list)  # 返回e的x次方
        t_list = t.view(-1)  # 让 t reshape为一行
        t_E = torch.gt(t_list, 0)

        y_pred_cumsum = torch.cumsum(y_pred_exp, dim=0)
        y_pred_cumsum_log = torch.log(y_pred_cumsum)  # 取对数



        loss1 = -torch.sum(y_pred_list.mul(t_E))  # 点乘运算
        loss2 = torch.sum(y_pred_cumsum_log.mul(t_E))

        loss = (loss1 + loss2) / torch.sum(t_E)
        return loss

def train_test_split_data(labels_txt, root_dir, KFlod, Fold, seed):
    """

    :param KFlod: fold
    :param seed:
    :param root_dir: 图像数据的储存位置
    :return: train_data,train_label,test_data,test_label
    """
    labels = []
    image_paths = []
    with open(labels_txt, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip("\n")
            img, label = line.split("\t")
            image_paths.append(os.path.join(root_dir, img))
            labels.append(label)

    k_fold = KFold(n_splits=KFlod, shuffle=True, random_state=seed)

    num = 0

    for train_indices, test_indices in k_fold.split(labels):
        train_data = [image_paths[i] for i in train_indices]
        train_label = [labels[i] for i in train_indices]
        test_data = [image_paths[i] for i in test_indices]
        test_label = [labels[i] for i in test_indices]
        if num == Fold:
            break
        num += 1

    print("Training Data:", len(train_data))
    print("Testing Data:", len(test_data))
    print("-" * 10)
    return train_data, train_label, test_data, test_label

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

class DRO_Loss(nn.Module):
    def __init__(self, temperature, tau_plus, batch_size, beta, estimator, N=1.2e6):
        super(DRO_Loss, self).__init__()

        self.temperature = temperature
        self.tau_plus = tau_plus
        self.batch_size = batch_size
        self.beta = beta
        self.estimator = estimator

    def forward(self, out_1, out_2, index=None, labels=None):
        device = out_1.device

        if self.estimator == "easy":

            out = torch.cat([out_1, out_2], dim=0)
            neg_ = torch.mm(out, out.t().contiguous())
            neg = torch.exp(neg_ / self.temperature)
            old_neg = neg.clone()
            mask = get_negative_mask(self.batch_size).to(device)
            neg = neg.masked_select(mask).view(2 * self.batch_size, -1)

            pos_ = torch.sum(out_1 * out_2, dim=-1)
            pos = torch.exp(pos_ / self.temperature)
            pos = torch.cat([pos, pos], dim=0)
            Ng = neg.sum(dim=-1)

            loss = (- torch.log(pos / (pos + Ng))).mean()

            return loss, None

        elif self.estimator == "HCL":

            batch_size = out_1.size(0)
            out = torch.cat([out_1, out_2], dim=0)
            neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
            old_neg = neg.clone()
            mask = get_negative_mask(batch_size).to(device)
            neg = neg.masked_select(mask).view(2 * batch_size, -1)

            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
            pos = torch.cat([pos, pos], dim=0)

            N = batch_size * 2 - 2
            imp = (self.beta * neg.log()).exp()
            reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
            Ng = (-self.tau_plus * N * pos + reweight_neg) / (1 - self.tau_plus)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.temperature))

            loss = (- torch.log(pos / (pos + Ng))).mean()

            return loss, None

        elif self.estimator == "a_cl":
            representations = torch.cat([out_1, out_2], dim=0)
            similarity_matrix = self.similarity_function(representations, representations)

            l_pos = torch.diag(similarity_matrix, self.batch_size)
            r_pos = torch.diag(similarity_matrix, -self.batch_size)

            positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

            negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

            dist_sqr = negatives - positives
            r_neg = 1 - negatives
            r_pos = 1 - positives
            w = r_neg.detach().pow(self.tau_plus)
            w = (-w / self.temperature).exp()
            w_Z = w.sum(dim=1, keepdim=True)
            w = w / (w_Z)
            loss = (w * dist_sqr).sum(dim=1).mean()
            return loss, None

        elif self.estimator == "a_cl2":
            representations = torch.cat([out_1, out_2], dim=0)
            similarity_matrix = self.similarity_function(representations, representations)

            l_pos = torch.diag(similarity_matrix, self.batch_size)
            r_pos = torch.diag(similarity_matrix, -self.batch_size)

            positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

            negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

            dist_sqr = negatives - positives
            r_neg = 1 - negatives
            r_pos = 1 - positives
            w = r_neg.detach().pow(self.tau_plus)
            w = (-w / self.temperature).exp()

            w_pos = w.sum(dim=1, keepdim=True)
            loss = (w_pos * r_pos - (w * r_neg).sum(dim=1)).mean()
            return loss, w

        elif self.estimator == "adnce":
            out = torch.cat([out_1, out_2], dim=0)
            neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
            old_neg = neg.clone()
            mask = get_negative_mask(self.batch_size).to(device)
            neg = neg.masked_select(mask).view(2 * self.batch_size, -1)
            # pos score
            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
            pos = torch.cat([pos, pos], dim=0)

            N = self.batch_size * 2 - 2
            import math
            mu = self.tau_plus
            sigma = self.beta
            weight = 1. / (sigma * math.sqrt(2 * math.pi)) * torch.exp(
                - (neg.log() * self.temperature - mu) ** 2 / (2 * math.pow(sigma, 2)))
            weight = weight / weight.mean(dim=-1, keepdim=True)
            # loss compute
            Ng = torch.sum(neg * weight.detach(), dim=1)
            loss = (- torch.log(pos / (pos + Ng))).mean()
            return loss, weight

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
