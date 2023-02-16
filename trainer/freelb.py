from .base import *


class FreeLBTrainer:
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
        adv_max_norm=1e-1
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

    def _inner_update(self, delta):
        delta_grad = delta.grad.clone().detach()
        _shape = None
        if delta.dim() > 3:
            # e.g. multi-choice
            _shape = delta.shape
            delta, delta_grad = delta.view(-1, _shape[-2], _shape[-1]), delta_grad.view(-1, _shape[-2], _shape[-1])

        grad_norm = torch.norm(delta_grad.view(delta_grad.shape[0], -1), dim=-1, p="fro")
        grad_norm = torch.clamp(grad_norm, min=1e-8).view(-1, 1, 1)
        self.delta = (delta + self.adv_lr * delta_grad / grad_norm).detach()

        delta_norm = torch.norm(delta.view(delta.shape[0], -1), dim=-1, p="fro").detach()
        clip_mask = (delta_norm > self.adv_max_norm).to(self.delta)
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

            delta = torch.zeros_like(inputs_embeds, requires_grad=True)
            for j in range(self.adv_steps):
                inputs_embeds = inputs_embeds + delta
                if self.fp16:
                    with autocast():
                        outputs = self.model(inputs_embeds=inputs_embeds,
                                             attention_mask=input_mask,
                                             token_type_ids=segment_ids,
                                             labels=labels)
                else:
                    outputs = self.model(inputs_embeds=inputs_embeds,
                                         attention_mask=input_mask,
                                         token_type_ids=segment_ids,
                                         labels=labels)
                loss = outputs[0].mean()

                loss = loss / self.adv_steps
                if self.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                train_loss += loss.item()
                if j == self.adv_steps - 1:
                    break

                self._inner_update(delta)
                delta.requires_grad_()
                inputs_embeds = self.word_embeddings(input_ids)

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
