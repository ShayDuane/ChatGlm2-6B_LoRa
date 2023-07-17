from datasets import load_from_disk
from transformers import (
    AutoModel,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    HfArgumentParser
)
import torch
from peft import TaskType, LoraConfig, get_peft_model
import os
import numpy as np
from dataclasses import field, dataclass
from typing import Union, List, Optional
from loguru import logger
import argparse

@dataclass
class FinetuneArguments():
    dataset_path: str = field(default='data/Med_datasets.jsonl')
    model_path: str = field(default='THUDM/chatglm2-6b')
    label_pad_token_id: int = field(default=-100)
    load_in_4bit: bool = field(default=True)
    bnb_4bit_quant_type: str = field(default='nf4')
    bnb_4bit_compute_dtype: str = field(default='float32')
    seed: int = field(default=42)
    resume_from_checkpoint: Union[str, bool] = field(default=None)
    final_model_path: str = field(default='QLora_Adapter_THUDM_chatglm2-6b/finally_adapter')

@dataclass
class LoraArguments():
    target_modules: Union[List[int], str] = field(default='query_key_value')
    r: int = field(default=12)
    lora_alpha: int = field(default=12)
    lora_dropout: float = field(default=0.05)
    bias: str = field(default='none')
    inference_mode: bool = field(default=False)
    layers_to_transform: List[int] = field(default=None)
    layers_pattern: str = field(default=None)
    Adapter_name: str = field(default='LoraAdapter')


class LoraTrainer(Trainer):

     def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
       # only save Lora adapter
        if output_dir is None:
            output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def parse_args():

    parser = argparse.ArgumentParser(description='ChatGlm2-6B_QLoRa-Args-Path')
    parser.add_argument('--finetune_args_path', type=str, default='args_file/finetune_args.json', help='path of finetune_args json file')
    parser.add_argument('--train_args_path', type=str, default='args_file/train_args.json', help='path of train_args json file')
    parser.add_argument('--lora_args_path', type=str, default='args_file/lora_args.json', help='path of lora_args json file')
    
    return parser.parse_args()




def train(args_path):

    finetune_args = HfArgumentParser(FinetuneArguments)
    finetune_args, = finetune_args.parse_json_file(json_file=args_path.finetune_args_path)
    logger.info(f"current_finetune_args--->{finetune_args}")

    train_args = HfArgumentParser(TrainingArguments)
    train_args, = train_args.parse_json_file(json_file=args_path.train_args_path)
    logger.info(f"current_train_args--->{train_args}")

    lora_args = HfArgumentParser(LoraArguments)
    lora_args, = lora_args.parse_json_file(json_file=args_path.lora_args_path)
    logger.info(f"current_lora_args--->{lora_args}")
    


    logger.info(f"from {finetune_args.dataset_path} loading datasets and tokenize datasets")
    datasets = load_from_disk(dataset_path=finetune_args.dataset_path)
    
    tokenizer = AutoTokenizer.from_pretrained(finetune_args.model_path, trust_remote_code=True)

    
    def tokenize_dataset(data, label_pad_token_id=finetune_args.label_pad_token_id):
        instruction = data['context']
        target = data['target']
        instruction_ids = tokenizer.encode(instruction, add_special_tokens=True)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        input_ids = instruction_ids + target_ids + [tokenizer.eos_token_id]
        labels = ([label_pad_token_id] * len(instruction_ids) + target_ids
                + [tokenizer.eos_token_id])
        return {'input_ids': input_ids, 'labels': labels}
    


    remove_columns_train = datasets['train'].column_names
    datasets_train = datasets['train'].map(tokenize_dataset,
                                        remove_columns=remove_columns_train)

    remove_columns_valid = datasets['valid'].column_names
    datasets_valid = datasets['valid'].map(tokenize_dataset,
                                        remove_columns=remove_columns_valid)
    logger.info(f"datasets_train--->{datasets_train.select(range(2))[0]}")

    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=finetune_args.load_in_4bit,
        bnb_4bit_quant_type=finetune_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=finetune_args.bnb_4bit_compute_dtype
    )


    logger.info(f"from {finetune_args.model_path} loading model")
    model = AutoModel.from_pretrained(finetune_args.model_path,
                                    trust_remote_code=True,
                                    device_map='auto',
                                    quantization_config=quantization_config,
                                    )


    logger.info("prepare model for training")
    for name, param in model.named_parameters():
        # freeze model parameters
        param.requires_grad = False

    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    # For backward compatibility
    model.enable_input_require_grads()
    # Enable gradient checkpoint
    model.gradient_checkpointing_enable()
    #`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`
    model.config.use_cache = False
    #but during inference make sure to set it back to True



    Lora_config = LoraConfig(
        r=lora_args.r,
        target_modules=lora_args.target_modules,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        inference_mode=lora_args.inference_mode,
        task_type=TaskType.CAUSAL_LM,
        layers_to_transform=lora_args.layers_to_transform,
        layers_pattern=lora_args.layers_pattern
    )
    logger.info(f"Lora_config--->{Lora_config}")


    model = get_peft_model(model, Lora_config, lora_args.Adapter_name)
    logger.info(f"lora_model--->{model}")

    logger.info("print trainable parameters")
    model.print_trainable_parameters()


    data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            pad_to_multiple_of=8,
            padding=True,
    )

    def compute_metrics(loss):
        loss_mean = loss.mean()
        perplexity = np.exp2(loss_mean)
        return {"perplexity": perplexity}

    

    

    trainer = LoraTrainer(
        model=model,
        args=train_args,
        train_dataset=datasets_train,
        eval_dataset=datasets_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )


    resume_from_checkpoint = finetune_args.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        if os.path.exists(resume_from_checkpoint):
            logger.info(f'Restarting from {resume_from_checkpoint}')
            model.load_adapter(os.path.join(resume_from_checkpoint, lora_args.Adapter_name), lora_args.Adapter_name)
        else:
            raise Exception(f'{resume_from_checkpoint} is not a correct path!')


    logger.info(f"start training from {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


    logger.info(f"training finished, save model to {finetune_args.final_model_path}")
    trainer.model.save_pretrained(finetune_args.final_model_path)
    torch.save(trainer.args, os.path.join(finetune_args.final_model_path, "training_args.bin"))


if __name__ == '__main__':
    args_path = parse_args()
    train(args_path)