from .base import *


class SMARTTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        max_train_steps,
        gradient_accumulation_steps=1,
        fp16=False,
        adv_steps=2,
        adv_lr=1e-3,
        adv_max_norm=1e-6,
        adv_temp=2e-2,
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

    def step(self, input_data):
        self.model.train()
        train_loss = 0
        train_step = 0
        for step, batch in enumerate(tqdm(input_data, desc="Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, labels = batch

            inputs_embeds = self.word_embeddings(input_ids)

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
            logits = outputs[1]

            delta = torch.randn_like(inputs_embeds, requires_grad=True) * self.adv_init_var
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
                logits_ptb = outputs[1]

                if j == self.adv_steps - 1:
                    smo_loss = kl_loss(logits_ptb, logits)
                    break

                smo_loss = kl_loss(logits_ptb, logits.detach())
                delta_grad, = torch.autograd.grad(smo_loss, delta)
                delta = delta + self.adv_lr * delta_grad
                delta_norm = torch.norm(delta, p=float("inf"), dim=-1, keepdim=True)
                delta = delta / (delta_norm + self.adv_max_norm)
                delta = delta.detach().requires_grad_()

            loss = loss + self.adv_temp * smo_loss

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
