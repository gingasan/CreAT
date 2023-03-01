from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
from tqdm import tqdm


def kl_loss(p, q):
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    return loss_fct(torch.log_softmax(p, -1), torch.softmax(q, -1)) + loss_fct(torch.log_softmax(q, -1), torch.softmax(p, -1))


def cos_loss(p, q):
    loss_fct = torch.nn.CosineSimilarity(dim=-1)
    return torch.sum(loss_fct(p, q), -1).mean()


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        max_train_steps,
        gradient_accumulation_steps=1,
        fp16=False
    ):
        self.model = model
        self.model_uw = model.module if hasattr(model, "module") else model
        self.device = model.device
        self.fp16 = fp16
        if self.fp16:
            self.scaler = GradScaler()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_train_steps = max_train_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_step = 0

    def step(self, input_data):
        self.model.train()
        train_loss = 0
        train_step = 0
        for step, batch in enumerate(tqdm(input_data, desc="Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, labels = batch

            if self.fp16:
                with autocast():
                    outputs = self.model(input_ids=input_ids,
                                         attention_mask=input_mask,
                                         token_type_ids=segment_ids,
                                         labels=labels)
            else:
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=input_mask,
                                     token_type_ids=segment_ids,
                                     labels=labels)
            loss = outputs[0].mean()

            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            train_loss += loss.item()
            train_step += 1
            if (step + 1) % self.gradient_accumulation_steps == 0 or step == len(input_data) - 1:
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                self.global_step += 1

            if self.global_step >= self.max_train_steps:
                break

        return train_loss, train_step
