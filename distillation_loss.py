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

        # Create a combined mask: Speech mask AND not padding (-100)
        if speech_token_mask is not None:
            shift_mask = speech_token_mask[..., 1:].contiguous().view(-1).bool()
            valid_mask = shift_mask & (shift_labels != -100)
        else:
            valid_mask = shift_labels != -100

        # Filter student once
        s_logits = shift_student[valid_mask]
        l_masked = shift_labels[valid_mask]

        if s_logits.size(0) == 0:
            return (
                torch.tensor(0.0, device=student_logits.device),
                torch.tensor(0.0, device=student_logits.device),
                torch.tensor(0.0, device=student_logits.device),
                torch.tensor(0.0, device=student_logits.device),
            )

        # Handle Teacher Logits (either full or sparse)
        if teacher_logits is not None:
            shift_teacher = (
                teacher_logits[..., :-1, :]
                .contiguous()
                .view(-1, teacher_logits.size(-1))
                .detach()
            )
            t_logits = shift_teacher[valid_mask]

            # Distillation Loss (KL Divergence - Dense)
            soft_t = F.softmax(t_logits / self.temperature, dim=-1)
            log_soft_s = F.log_softmax(s_logits / self.temperature, dim=-1)
            distill_loss = self.kl_div(log_soft_s, soft_t) * (self.temperature**2)

            # Teacher Performance Monitoring
            teacher_task_loss = F.cross_entropy(t_logits, l_masked)

        elif teacher_top_k_v is not None and teacher_top_k_i is not None:
            # SPARSE DISTILLATION: No full-vocab tensor allocation
            K = teacher_top_k_v.size(-1)

            # Extract and move sparse data, then filter by valid_mask
            v_valid = (
                teacher_top_k_v[..., :-1, :]
                .contiguous()
                .view(-1, K)[valid_mask]
                .to(student_logits.device)
            )
            i_valid = (
                teacher_top_k_i[..., :-1, :]
                .contiguous()
                .view(-1, K)[valid_mask]
                .to(student_logits.device)
            )

            # Teacher distribution (renormalized over Top-K at current temperature)
            soft_t = F.softmax(v_valid.float() / self.temperature, dim=-1)
            log_soft_t = F.log_softmax(v_valid.float() / self.temperature, dim=-1)

            # Student distribution (full vocab)
            log_soft_s_all = F.log_softmax(s_logits / self.temperature, dim=-1)

            # Gather student logprobs at teacher's Top-K indices
            log_soft_s_gathered = log_soft_s_all.gather(-1, i_valid.long())

            # Sparse KL: \sum P_teacher * (log P_teacher - log Q_student)
            # We use sum reduction and divide by number of tokens manually
            distill_loss = (
                (soft_t * (log_soft_t - log_soft_s_gathered)).sum(dim=-1).mean()
            ) * (self.temperature**2)

            # Teacher Performance Monitoring (Approximate: only if label is in Top-K)
            # Find if ground truth label is in the Top-K indices
            target_pos = (i_valid.long() == l_masked.unsqueeze(-1)).nonzero(
                as_tuple=True
            )
            if target_pos[0].size(0) > 0:
                # Get the teacher logprob for the target labels that were found in Top-K
                v_target = v_valid[target_pos[0], target_pos[1]]
                teacher_task_loss = -v_target.mean()
            else:
                teacher_task_loss = torch.tensor(0.0, device=v_valid.device)
        else:
            raise ValueError("Either teacher_logits or top_k must be provided")

        # Task Loss (Standard Cross Entropy)
        task_loss = F.cross_entropy(s_logits, l_masked)

        # Combined loss
        total_loss = self.alpha * task_loss + (1 - self.alpha) * distill_loss

        return total_loss, task_loss, distill_loss, teacher_task_loss
