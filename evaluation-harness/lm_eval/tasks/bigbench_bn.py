"""
Measuring Massive Multitask Language Understanding
https://arxiv.org/pdf/2009.03300.pdf

The Hendryck's Test is a benchmark that measured a text model’s multitask accuracy.
The test covers 57 tasks including elementary mathematics, US history, computer
science, law, and more. To attain high accuracy on this test, models must possess
extensive world knowledge and problem solving ability. By comprehensively evaluating
the breadth and depth of a model’s academic and professional understanding,
Hendryck's Test can be used to analyze models across many tasks and to identify
important shortcomings.

Homepage: https://github.com/hendrycks/test
"""
from lm_eval.base import MultipleChoiceTask, Task, rf
from lm_eval.metrics import mean, acc_all, metric_max_over_ground_truths, yesno, skip
from lm_eval.utils import general_detokenize, InstructionTemplates
import numpy as np
import re
import datasets

_CITATION = """
@article{hendryckstest2021,
    title={Measuring Massive Multitask Language Understanding},
    author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2021}
}
"""

# Tasks
# causal_judgement-en-bn.csv  disambiguation_qa-en-bn.csv  logical_deductions_five-en-bn.csv   logical_deductions_three-en-bn.csv  penguins_in_a_table-en-bn.csv              snarks-en-bn.csv                temporal_sequences-en-bn.csv
# date_understanding.csv      formal_fallacies-en-bn.csv   logical_deductions_seven-en-bn.csv  navigate-en-bn.csv                  reasoning_about_colored_objects-en-bn.csv  sports_understanding-en-bn.csv  web_of_lies-en-bn.csv

TASK_TO_FILE = {
    'causal_judgement': 'causal_judgement-en-bn.csv',
    'date_understanding': 'date_understanding.csv',
    'disambiguation_qa': 'disambiguation_qa-en-bn.csv',
    'formal_fallacies': 'formal_fallacies-en-bn.csv',
    'logical_deductions_five': 'logical_deductions_five-en-bn.csv',
    'logical_deductions_seven': 'logical_deductions_seven-en-bn.csv',
    'logical_deductions_three': 'logical_deductions_three-en-bn.csv',
    'navigate': 'navigate-en-bn.csv',
    'penguins_in_a_table': 'penguins_in_a_table-en-bn.csv',
    'reasoning_about_colored_objects': 'reasoning_about_colored_objects-en-bn.csv',
    'snarks': 'snarks-en-bn.csv',
    'temporal_sequences': 'temporal_sequences-en-bn.csv',
    'sports_understanding': 'sports_understanding-en-bn.csv',
    'web_of_lies': 'web_of_lies-en-bn.csv',
}


def create_all_tasks():
    tasks = {f'bbh_{task}_en': create_task_gen(
        task, 'en') for task in TASK_TO_FILE.keys()}
    tasks.update({f'bbh_{task}_bn': create_task_gen(task, 'bn')
                 for task in TASK_TO_FILE.keys()})
    return tasks


def create_task_gen(task, lang):
    class BBHGenTest(BBHGen):
        def __init__(self):
            super().__init__(task, lang)

    return BBHGenTest


class BBHGen(Task):
    ANS_RE = [r"\([A-Z]\)", r"[A-Z]\.", r"[A-Z]"]
    ANS_RE_YES_NO = [r'Yes|No', r'yes|no']
    ANS_RE_VALID = [r'valid|invalid|Valid|Invalid']

    ORCA_SYSTEM = (
        "You are an AI assistant. User will you give you a task. "
        "Your goal is to complete the task as faithfully as you can. "
        "While performing the task think step-by-step and justify your steps."
    )

    ORCA_INSTRUCTION = (
        "Choose an answer from the choices provided. All questions are answerable.\n"
        "You must choose an option. If all options seem wrong, choose the option that is closest to the answer.\n"
        "At the end output\n"
        "###Final answer: {{answer choice}}\n\n"
    )

    VERSION = 0
    DATASET_PATH = "../data/BBH-Bengali/"

    def __init__(self, task, lang):
        assert lang in ['en', 'bn']
        self.TASK = task
        self.DATASET_NAME = lang
        super().__init__()

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        file_name = TASK_TO_FILE[self.TASK]
        self.dataset = datasets.load_dataset(
            'csv', data_files=self.DATASET_PATH + file_name)['train']

    def _process_docs(self, doc):
        if self.DATASET_NAME == 'en':
            question = doc['input_en']
            choices = doc['options_en']

        elif self.DATASET_NAME == 'bn':
            question = doc['input_bn']
            choices = doc['options_bn']

        answer = doc['target_en']

        return {
            "question": question,
            "choices": choices,
            "answer": answer,
        }

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return map(self._process_docs, self.dataset)

    def format_example(self, doc, instruction_template=None):
        """
        Question: <prompt>
        Options:
        A. <choice1>
        B. <choice2>
        C. <choice3>
        D. <choice4>
        """
        prompt = doc["question"] + "\n" + doc["choices"]
        if instruction_template != "base":
            prompt += "\nStep-by-step answer:"
        else:
            prompt += "\nAnswer:"
        return prompt

    def doc_to_text(self, doc, instruction_template=None):
        return self.doc_to_text_with_instruction(doc, instruction_template)

    def doc_to_text_with_instruction(self, doc, instruction_template):
        instruction = self.format_example(doc, instruction_template)
        user_message = self.ORCA_INSTRUCTION + instruction
        if "orca" in instruction_template:
            template = InstructionTemplates.get_template(instruction_template)
            return template.format(
                system_message=self.ORCA_SYSTEM,
                user_message=user_message,
            )
        elif instruction_template == "bactrian":
            template = InstructionTemplates.get_template(instruction_template)
            user_message = self.ORCA_SYSTEM + "\n" + user_message
            return template.format(
                user_message=user_message,
            )
        elif instruction_template == "base":
            return user_message
        else:
            raise ValueError(
                f"Unknown instruction template: {instruction_template}")

    def doc_to_target(self, doc, instruction_template=None):
        return self.doc_to_target_with_instruction(doc, instruction_template)

    def doc_to_target_with_instruction(self, doc, instruction_template):
        return doc['choices']

    def construct_requests(self, doc, ctx, instruction_template=None):
        completion = rf.greedy_until(
            ctx, {"until": ['<|im_end|>']})
        return completion

    def _extract_answer(self, completion, patterns):
        for pattern in patterns:
            match = re.findall(pattern, completion)
            if match:
                return match[-1]
        return ""

    def _is_correct(self, completion, answer, patterns):
        gold = answer
        pred = self._extract_answer(completion, patterns)
        # strip parenthesis
        pred = pred.strip('(').strip(')')
        gold = gold.strip('(').strip(')')

        print(f"pred: {pred}, gold: {gold}")

        return pred == gold

    def process_results(self, doc, results):
        print(f'results: {results}')
        completion = results[0]
        correct_choice = doc["answer"]

        if correct_choice in ['Yes', 'No']:
            patterns = self.ANS_RE_YES_NO
        elif correct_choice in ['valid', 'invalid']:
            patterns = self.ANS_RE_VALID
        else:
            patterns = self.ANS_RE

        # assert re.findall(patterns[0], doc["answer"]
        #                   ), f"answer: {doc['answer']}"
        return {"acc": self._is_correct(completion, correct_choice, patterns)}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}
