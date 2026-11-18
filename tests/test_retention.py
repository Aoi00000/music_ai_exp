import unittest
import torch
from band2tab.retention import retention_loss


class RetentionTests(unittest.TestCase):
    def test_padding_is_ignored_and_teacher_is_detached(self):
        student=torch.zeros(1,1,3,5,requires_grad=True);teacher=torch.randn(1,1,3,5,requires_grad=True)
        labels=torch.tensor([[[1,2,0]]]);kl,ce=retention_loss(student,teacher,labels)
        (kl+ce).backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad[:,:,:2].abs().sum()),0)
        self.assertEqual(float(student.grad[:,:,2].abs().sum()),0)

    def test_matching_logits_have_zero_kl_and_empty_batch_is_finite(self):
        logits=torch.randn(1,1,4,5);labels=torch.tensor([[[1,2,3,0]]])
        kl,ce=retention_loss(logits,logits,labels)
        self.assertAlmostEqual(float(kl),0,places=6)
        kl,ce=retention_loss(logits,logits,labels*0)
        self.assertEqual(float(kl+ce),0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'shapes'):
            retention_loss(torch.zeros(1,1,2,5),torch.zeros(1,1,3,5),torch.ones(1,1,2,dtype=torch.long))


if __name__=='__main__':unittest.main()
