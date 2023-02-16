from torch.cuda.amp import autocast, GradScaler
import torch
from tqdm import tqdm


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
        gradient_accumulation_steps=0,
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


class CreATTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        max_train_steps,
        gradient_accumulation_steps=0,
        fp16=False,
        adv_steps=2,
        adv_lr=1e-1,
        adv_max_norm=1e-1,
        adv_temp=1.0,
        adv_init_var=1e-5
    ):
        self.model = model
        self.model_uw = model.module if hasattr(model, "module") else model
        self.word_embeddings = getattr(self.model_uw, self.model_uw.config.model_type).embeddings.word_embeddings
        self.device = model.device
        self.fp16 = fp16
        if self.fp16:
            self.scaler = GradScaler()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_train_steps = max_train_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_step = 0

        self.adv_steps = adv_steps
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_temp = adv_temp
        self.adv_init_var = adv_init_var

    def _inner_update(self, delta, loss):
        delta_grad, = torch.autograd.grad(loss, delta)
        _shape = None
        if delta.dim() > 3:
            # e.g. multi-choice
            _shape = delta.shape
            delta, delta_grad = delta.view(-1, _shape[-2], _shape[-1]), delta_grad.view(-1, _shape[-2], _shape[-1])

        grad_norm = torch.norm(delta_grad.view(delta_grad.shape[0], -1), dim=-1, p="fro")
        grad_norm = torch.clamp(grad_norm, min=1e-8).view(-1, 1, 1)
        delta = (delta + self.adv_lr * delta_grad / grad_norm).detach()

        delta_norm = torch.norm(delta.view(delta.shape[0], -1), dim=-1, p="fro").detach()
        clip_mask = (delta_norm > self.adv_max_norm).to(delta)
        clip_weights = self.adv_max_norm / delta_norm * clip_mask + (1 - clip_mask)
        delta = (delta * clip_weights.view(-1, 1, 1)).detach()

        if _shape is not None:
            delta = delta.view(_shape)

        return delta

    def step(self, input_data):
        self.model.train()
        train_loss = 0
        train_step = 0
        for step, batch in enumerate(tqdm(input_data, desc="Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, labels = batch

            inputs_embeds = self.word_embeddings(input_ids)
            extended_input_mask = input_mask.view(-1, input_mask.size(-1)).unsqueeze(-1)

            if self.fp16:
                with autocast():
                    outputs = self.model(inputs_embeds=inputs_embeds,
                                         attention_mask=input_mask,
                                         token_type_ids=segment_ids,
                                         labels=labels,
                                         output_hidden_states=True)
            else:
                outputs = self.model(inputs_embeds=inputs_embeds,
                                     attention_mask=input_mask,
                                     token_type_ids=segment_ids,
                                     labels=labels,
                                     output_hidden_states=True)
            loss = outputs[0].mean()
            ctxr = outputs[-1][-1] * extended_input_mask

            delta = torch.randn_like(inputs_embeds, requires_grad=True) * self.adv_init_var
            for j in range(self.adv_steps):
                inputs_embeds = inputs_embeds + delta
                if self.fp16:
                    with autocast():
                        outputs = self.model(inputs_embeds=inputs_embeds,
                                             attention_mask=input_mask,
                                             token_type_ids=segment_ids,
                                             labels=labels,
                                             output_hidden_states=True)
                else:
                    outputs = self.model(inputs_embeds=inputs_embeds,
                                         attention_mask=input_mask,
                                         token_type_ids=segment_ids,
                                         labels=labels,
                                         output_hidden_states=True)
                loss_ptb = outputs[0].mean()
                ctxr_ptb = outputs[-1][-1] * extended_input_mask

                if j == self.adv_steps - 1:
                    break

                loss_ptb = loss_ptb - cos_loss(ctxr_ptb, ctxr.detach()) * self.adv_temp
                delta = self._inner_update(delta, loss_ptb)
                delta.requires_grad_()

            loss = 0.5 * (loss + loss_ptb)

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
