# Contextualized representation-Adversarial Training

This repo is for the ICLR 2023 paper [Toward Adversarial Training on Contextualized Language Representation](https://openreview.net/pdf?id=xZD10GhCvM).



## Trainer

We implement a number of adversarial training algorithms in `trainer`, e.g. FreeLBTrainer, SMARTTrainer.

The current version supports huggingface BERT, RoBERTa, DeBERTa, ALBERT, etc.

**CreAT:**

```python
from trainer.creat import CreATTrainer

trainer = CreATTrainer(model, optimizer, scheduler, max_train_steps=10000, fp16=True)

for epoch in trange(3):
  train_loss, train_step = trainer.step(train_dataloader)
  global_step = trainer.global_step
```

**SMART:**

```python
from trainer.smart import SMARTTrainer

trainer = SMARTTrainer(model, optimizer, scheduler, max_train_steps=10000, fp16=True)

for epoch in trange(3):
  train_loss, train_step = trainer.step(train_dataloader)
  global_step = trainer.global_step
```

**R3F:**

```python
from trainer.r3f import R3FTrainer

trainer = R3FTrainer(model, optimizer, scheduler, max_train_steps=10000, fp16=True)

for epoch in trange(3):
  train_loss, train_step = trainer.step(train_dataloader)
  global_step = trainer.global_step
```

**FreeLB:**

```python
from trainer.freelb import FreeLBTrainer

trainer = FreeLBTrainer(model, optimizer, scheduler, max_train_steps=10000, fp16=True)

for epoch in trange(3):
  train_loss, train_step = trainer.step(train_dataloader)
  global_step = trainer.global_step
```

**Standard training:**

```python
from trainer.base import Trainer

trainer = Trainer(model, optimizer, scheduler, max_train_steps=10000, fp16=True)

for epoch in trange(3):
  train_loss, train_step = trainer.step(train_dataloader)
  global_step = trainer.global_step
```

