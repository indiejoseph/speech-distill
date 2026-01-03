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
        # Shift logits and labels for causal LM: logits[i] predicts labels[i+1]
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if speech_token_mask is not None:
            # shift_mask[i] == 1 means shift_labels[i] (which is labels[i+1]) is a speech token
            shift_mask = speech_token_mask[..., 1:].contiguous()
            mask_flat = shift_mask.view(-1).bool()

            # Apply mask to shifted logits and labels
            student_logits_masked = shift_student_logits.view(
                -1, shift_student_logits.size(-1)
            )[mask_flat]
            teacher_logits_masked = shift_teacher_logits.view(
                -1, shift_teacher_logits.size(-1)
            )[mask_flat]
            labels_masked = shift_labels.view(-1)[mask_flat]
        else:
            # Use entire sequence (shifted)
            student_logits_masked = shift_student_logits.view(
                -1, shift_student_logits.size(-1)
            )
            teacher_logits_masked = shift_teacher_logits.view(
                -1, shift_teacher_logits.size(-1)
            )
            labels_masked = shift_labels.view(-1)

        # Soften logits with temperature
        soft_teacher_probs = F.softmax(teacher_logits_masked / self.temperature, dim=-1)
        soft_student_log_probs = F.log_softmax(
            student_logits_masked / self.temperature, dim=-1
        )

        # KL Divergence loss
        distill_loss = self.kl_div(soft_student_log_probs, soft_teacher_probs) * (
            self.temperature**2
        )

        # Standard Cross Entropy loss
        # Filter out padding/ignored tokens (labels == -100)
        ce_mask = labels_masked != -100
        if ce_mask.sum() > 0:
            student_logits_ce = student_logits_masked[ce_mask]
            teacher_logits_ce = teacher_logits_masked[ce_mask]
            labels_ce = labels_masked[ce_mask]
            task_loss = F.cross_entropy(student_logits_ce, labels_ce)
            teacher_task_loss = F.cross_entropy(teacher_logits_ce, labels_ce)
        else:
            task_loss = torch.tensor(0.0).to(student_logits.device)
            teacher_task_loss = torch.tensor(0.0).to(student_logits.device)

        # Combined loss
        total_loss = self.alpha * task_loss + (1 - self.alpha) * distill_loss

        return total_loss, task_loss, distill_loss, teacher_task_loss
