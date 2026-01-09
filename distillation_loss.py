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

    def forward(
        self,
        student_logits,
        labels,
        teacher_logits=None,
        teacher_top_k_v=None,
        teacher_top_k_i=None,
        speech_token_mask=None,
    ):
        """
        student_logits: [batch_size, seq_len, vocab_size]
        teacher_logits: [batch_size, seq_len, vocab_size] (Optional if top_k provided)
        teacher_top_k_v: [batch_size, seq_len, K] (Optional)
        teacher_top_k_i: [batch_size, seq_len, K] (Optional)
        labels: [batch_size, seq_len]
        """
        # Causal shift: logits[i] predicts labels[i+1]
        shift_student = (
            student_logits[..., :-1, :].contiguous().view(-1, student_logits.size(-1))
        )
        shift_labels = labels[..., 1:].contiguous().view(-1)

        # Handle Teacher Logits (either full or sparse)
        if teacher_logits is not None:
            shift_teacher = (
                teacher_logits[..., :-1, :]
                .contiguous()
                .view(-1, teacher_logits.size(-1))
                .detach()
            )
        elif teacher_top_k_v is not None and teacher_top_k_i is not None:
            # Reconstruct sparse logprobs
            B_seq, T_seq, K = teacher_top_k_v.shape
            V = student_logits.size(-1)

            # Move sparse data to student device and shift
            v = (
                teacher_top_k_v[..., :-1, :]
                .contiguous()
                .view(-1, K)
                .to(student_logits.device)
            )
            i = (
                teacher_top_k_i[..., :-1, :]
                .contiguous()
                .view(-1, K)
                .to(student_logits.device)
            )

            # For KL divergence, we can often just use the Top-K as a sparse distribution
            # provided we handle the softmax correctly.
            # Here we reconstruct a sparse "pseudo-distribution"
            shift_teacher = torch.full(
                (v.size(0), V), -10000.0, device=v.device, dtype=v.dtype
            )
            shift_teacher.scatter_(-1, i.long(), v.float())  # Restore logprobs
        else:
            raise ValueError("Either teacher_logits or top_k must be provided")

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
