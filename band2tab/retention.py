"""Masked teacher retention; pseudo targets are never evaluation ground truth."""
import torch


def retention_loss(student,teacher,labels,temperature=2.):
    if student.shape!=teacher.shape or student.shape[:-1]!=labels.shape:raise ValueError('Retention tensor shapes differ')
    if temperature<=0:raise ValueError('Retention temperature must be positive')
    mask=labels!=0;t=temperature
    per_token=torch.nn.functional.kl_div((student.float()/t).log_softmax(-1),(teacher.detach().float()/t).softmax(-1),reduction='none').sum(-1)*t*t
    kl=(per_token*mask).sum()/mask.sum().clamp_min(1)
    ce=torch.nn.functional.cross_entropy(student.float().reshape(-1,student.shape[-1]),labels.reshape(-1),ignore_index=0,reduction='none').sum()/mask.sum().clamp_min(1)
    return kl,ce
