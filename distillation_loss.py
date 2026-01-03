import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels, speech_token_mask=None):
        """
        student_logits: [batch_size, seq_len, vocab_size]
        teacher_logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]
        speech_token_mask: [batch_size, seq_len] - binary mask where 1 = speech token positions
                          If None, use entire sequence. If provided, only compute loss on speech tokens.
        """
        # Causal shift: logits[i] predicts labels[i+1]
        shift_student = (
            student_logits[..., :-1, :].contiguous().view(-1, student_logits.size(-1))
        )
        shift_teacher = (
            teacher_logits[..., :-1, :]
            .contiguous()
            .view(-1, teacher_logits.size(-1))
            .detach()
        )
        shift_labels = labels[..., 1:].contiguous().view(-1)

        # Create a combined mask: Speech mask AND not padding (-100)
        if speech_token_mask is not None:
            shift_mask = speech_token_mask[..., 1:].contiguous().view(-1).bool()
            valid_mask = shift_mask & (shift_labels != -100)
        else:
            valid_mask = shift_labels != -100

        # Filter everything once
        s_logits = shift_student[valid_mask]
        t_logits = shift_teacher[valid_mask]
        l_masked = shift_labels[valid_mask]

        if s_logits.size(0) == 0:
            return (
                torch.tensor(0.0, device=student_logits.device),
                torch.tensor(0.0, device=student_logits.device),
                torch.tensor(0.0, device=student_logits.device),
                torch.tensor(0.0, device=student_logits.device),
            )

        # Distillation Loss (KL Divergence)
        soft_t = F.softmax(t_logits / self.temperature, dim=-1)
        log_soft_s = F.log_softmax(s_logits / self.temperature, dim=-1)
        distill_loss = self.kl_div(log_soft_s, soft_t) * (self.temperature**2)

        # Task Loss (Standard Cross Entropy)
        task_loss = F.cross_entropy(s_logits, l_masked)

        # Teacher Performance Monitoring (no gradients needed as t_logits is detached)
        teacher_task_loss = F.cross_entropy(t_logits, l_masked)

        # Combined loss
        total_loss = self.alpha * task_loss + (1 - self.alpha) * distill_loss

        return total_loss, task_loss, distill_loss, teacher_task_loss
