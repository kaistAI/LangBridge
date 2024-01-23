import argparse
import json
import logging
import os

import torch
from transformers import AutoTokenizer

from langbridge import LangBridgeModel

from lm_eval import tasks, evaluator, utils
from lm_eval.models.langbridge import LBSeq2SeqLM

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--enc_tokenizer", default='')
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None,
                        choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--instruction_template", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(
            args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    print('instruction_template:', args.instruction_template)

    model = LangBridgeModel.from_pretrained(
        args.checkpoint_path, torch_dtype=torch.bfloat16)
    model.to(args.device)

    try:
        enc_tokenizer = AutoTokenizer.from_pretrained(
            args.enc_tokenizer, use_fast=False)
    except:
        enc_tokenizer = AutoTokenizer.from_pretrained(
            args.enc_tokenizer, use_fast=True)

    try:
        lm_tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint_path, use_fast=False)
    except:
        lm_tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint_path, use_fast=True)

    if not enc_tokenizer.pad_token:
        enc_tokenizer.pad_token = enc_tokenizer.eos_token
    if not lm_tokenizer.pad_token:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token

    eval_model = LBSeq2SeqLM(
        model=model,
        enc_tokenizer=enc_tokenizer,
        lm_tokenizer=lm_tokenizer,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
    )

    results = evaluator.simple_evaluate(
        model=eval_model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
        instruction_template=args.instruction_template,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"{args.checkpoint_path} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
