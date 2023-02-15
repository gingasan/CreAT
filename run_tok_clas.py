from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import sklearn.metrics as mtc
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import SchedulerType, get_scheduler
from creat import Trainer, CreATTrainer


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, labels=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class ConllProcessor:
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    @staticmethod
    def _read_csv(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            sentence = []
            labels = []
            for line in f:
                if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
                    if len(sentence) > 0:
                        lines.append((sentence, labels))
                        sentence = []
                        labels = []
                    continue
                splits = line.split()
                sentence.append(splits[0])
                labels.append(splits[-1])
            if len(sentence) > 0:
                lines.append((sentence, labels))
            return lines

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for i, (sentence, labels) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, labels=labels))
        return examples


class WnutProcessor:
    """Processor for the WNUT-2017 data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B-corporation", "B-creative-work", "B-group", "B-location", "B-person", "B-product",
                "I-corporation", "I-creative-work", "I-group", "I-location", "I-person", "I-product"]

    @staticmethod
    def _read_csv(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            sentence = []
            labels = []
            for line in f:
                if len(line) == 0 or line[0] == "\n":
                    if len(sentence) > 0:
                        lines.append((sentence, labels))
                        sentence = []
                        labels = []
                    continue
                splits = line.split()
                sentence.append(splits[0])
                labels.append(splits[-1])
            if len(sentence) > 0:
                lines.append((sentence, labels))
            return lines

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for i, (sentence, labels) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, labels=labels))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for i, example in enumerate(examples):
        encoded_inputs = tokenizer(example.text_a,
                                   max_length=max_seq_length,
                                   padding="max_length",
                                   truncation=True,
                                   return_token_type_ids=True,
                                   is_split_into_words=True)

        labels = example.labels
        word_ids = encoded_inputs.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_map[labels[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        input_ids = encoded_inputs["input_ids"]
        input_mask = encoded_inputs["attention_mask"]
        segment_ids = encoded_inputs["token_type_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if i < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(tokens))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.labels, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids)
        )
    return features


class Metrics:
    @staticmethod
    def acc(predictions, labels):
        return mtc.accuracy_score(labels, predictions)

    @staticmethod
    def mcc(predictions, labels):
        return mtc.matthews_corrcoef(labels, predictions)

    @staticmethod
    def f1(predictions, labels, average="micro"):
        return mtc.f1_score(labels, predictions, average=average)


def main():
    parser = argparse.ArgumentParser()

    # Data config
    parser.add_argument("--data_dir", type=str, default="../data/ner/",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--task_name", type=str, default="CoNLL-03",
                        help="Name of the training task.")
    parser.add_argument("--load_model_path", type=str, default="bert-base-uncased",
                        help="Specific model path to load, e.g. bert-base-uncased.")
    parser.add_argument("--cache_dir", type=str, default="../cache/",
                        help="Directory to store the pre-trained language models downloaded from s3.")
    parser.add_argument("--output_dir", type=str, default="model/",
                        help="Directory to output predictions and checkpoints.")

    # Training config
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to evaluate on the dev set.")
    parser.add_argument("--eval_on", type=str, default="test",
                        help="Whether to evaluate on the test set.")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="A slow tokenizer will be used if passed.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", type=int, default=64,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides training epochs.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        help="Scheduler type for learning rate warmup.")
    parser.add_argument("--warmup_proportion", type=float, default=0.1,
                        help="Proportion of training to perform learning rate warmup for.")
    parser.add_argument("--weight_decay", type=float, default=0.,
                        help="L2 weight decay for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward pass.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use mixed precision.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")
    parser.add_argument("--creat", action="store_true",
                        help="To do CreAT if passed.")
    parser.add_argument("--adv_steps", type=int, default=2,
                        help="Inner ascent steps for AT.")
    parser.add_argument("--adv_lr", type=float, default=1e-1,
                        help="Step size for AT.")
    parser.add_argument("--adv_max_norm", type=float, default=1e-1,
                        help="Decision boundary for AT.")
    parser.add_argument("--adv_temp", type=float, default=1.0,
                        help="Temperature coefficient for AT.")

    args = parser.parse_args()

    processors = {
        "conll-03": ConllProcessor,
        "wnut-17": WnutProcessor,
    }

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, "Unsupported", args.fp16))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        torch.save(args, os.path.join(args.output_dir, "train_args.bin"))
    elif args.do_test:
        torch.save(args, os.path.join(args.output_dir, "test_args.bin"))

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                              do_lower_case=args.do_lower_case,
                                              cache_dir=cache_dir,
                                              use_fast=not args.use_slow_tokenizer,
                                              add_prefix_space=True)

    if args.do_train:
        train_examples = processor.get_train_examples(os.path.join(args.data_dir, task_name))
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        model = AutoModelForTokenClassification.from_pretrained(args.load_model_path,
                                                                num_labels=num_labels,
                                                                return_dict=True,
                                                                cache_dir=cache_dir)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_scheduler(name=args.lr_scheduler_type,
                                  optimizer=optimizer,
                                  num_warmup_steps=args.max_train_steps * args.warmup_proportion,
                                  num_training_steps=args.max_train_steps)
        
        if args.do_eval:
            if args.eval_on == "dev":
                eval_examples = processor.get_dev_examples(os.path.join(args.data_dir, task_name))
            else:
                eval_examples = processor.get_test_examples(os.path.join(args.data_dir, task_name))
            eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)

            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", args.max_train_steps)

        if args.creat:
            trainer = CreATTrainer(model, optimizer, scheduler, args.max_train_steps, args.gradient_accumulation_steps, args.fp16,
                                    args.adv_steps, args.adv_lr, args.adv_max_norm, args.adv_temp)
        else:
            trainer = Trainer(model, optimizer, scheduler, args.max_train_steps, args.gradient_accumulation_steps, args.fp16)

        global_step = 0
        for epoch in trange(int(args.num_train_epochs)):

            train_loss, train_step = trainer.step(train_dataloader)
            global_step = trainer.global_step

            model_to_save = model.module if hasattr(model, "module") else model
            output_model_file = os.path.join(args.output_dir, "{}_pytorch_model.bin".format(epoch))
            torch.save(model_to_save.state_dict(), output_model_file)

            if args.do_eval:
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                model.eval()
                eval_loss = 0
                num_eval_examples = 0
                eval_steps = 0
                all_predictions, all_labels = [], []
                for batch in tqdm(eval_dataloader, desc="Evaluation"):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids,
                                        attention_mask=input_mask,
                                        token_type_ids=segment_ids,
                                        labels=label_ids)
                        tmp_eval_loss = outputs[0]
                        logits = outputs[1]

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to("cpu").numpy()
                    eval_loss += tmp_eval_loss.mean().item()
                    tmp_predictions = np.argmax(logits, axis=2).reshape(-1).tolist()
                    tmp_labels = label_ids.reshape(-1).tolist()
                    all_predictions.extend([p for p, l in zip(tmp_predictions, tmp_labels) if l != -100])
                    all_labels.extend([l for l in tmp_labels if l != -100])
                    num_eval_examples += input_ids.size(0)
                    eval_steps += 1

                loss = train_loss / train_step
                eval_loss = eval_loss / eval_steps
                eval_acc = mtc.f1_score(all_labels,
                                        all_predictions,
                                        labels=list(range(1, num_labels)),
                                        average="micro") * 100

                result = {
                    "global_step": global_step,
                    "loss": loss,
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                }

                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, "a") as writer:
                    logger.info("***** Eval results *****")
                    writer.write(
                        "Epoch %s: global step = %s | loss = %.3f | eval score = %.2f | eval loss = %.3f\n"
                        % (str(epoch),
                           str(result["global_step"]),
                           result["loss"],
                           result["eval_acc"],
                           result["eval_loss"]))
                    for key in sorted(result.keys()):
                        logger.info("Epoch: %s,  %s = %s", str(epoch), key, str(result[key]))


if __name__ == "__main__":
    main()
