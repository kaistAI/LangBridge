"""
Language Models are Multilingual Chain-of-Thought Reasoners
https://arxiv.org/abs/2210.03057

Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems, proposed in the paper [Language models are multilingual chain-of-thought reasoners](http://arxiv.org/abs/2210.03057).

The same 250 problems from [GSM8K](https://arxiv.org/abs/2110.14168) are each translated via human annotators in 10 languages. The 10 languages are:
- Spanish
- French
- German
- Russian
- Chinese
- Japanese
- Thai
- Swahili
- Bengali
- Telugu

GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.

You can find the input and targets for each of the ten languages (and English) as `.tsv` files.
We also include few-shot exemplars that are also manually translated from each language in `exemplars.py`.

Homepage: https://github.com/google-research/url-nlp/tree/main/mgsm
"""
import re
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
import datasets
from lm_eval.utils import InstructionTemplates


_CITATION = """
@misc{cobbe2021training,
    title={Training Verifiers to Solve Math Word Problems},
    author={Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
    year={2021},
    eprint={2110.14168},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
@misc{shi2022language,
    title={Language Models are Multilingual Chain-of-Thought Reasoners},
    author={Freda Shi and Mirac Suzgun and Markus Freitag and Xuezhi Wang and Suraj Srivats and Soroush Vosoughi and Hyung Won Chung and Yi Tay and Sebastian Ruder and Denny Zhou and Dipanjan Das and Jason Wei},
    year={2022},
    eprint={2210.03057},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

ANS_RE = re.compile(r"(\-?\d+)")
INVALID_ANS = "[invalid]"


class MGSM(Task):
    VERSION = 0
    DATASET_PATH = "juletxara/mgsm"
    DATASET_NAME = None
    QUESTION = "Question:"
    ANSWER = "Step-by-Step Answer:"

    ORCA_SYSTEM = (
        "You are an AI assistant. User will you give you a task. "
        "Your goal is to complete the task as faithfully as you can. "
        "While performing the task think step-by-step and justify your steps."
    )

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = datasets.load_dataset(
            self.DATASET_PATH,
            self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )
        if self.DATASET_NAME == "en":
            return
        self.en_dataset = datasets.load_dataset(
            self.DATASET_PATH,
            "en",
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

        self.dataset['train'] = self.dataset['train'].remove_columns('answer')
        self.dataset['train'] = self.dataset['train'].add_column(
            'answer', self.en_dataset['train']['answer'])

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        raise NotImplementedError

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc, instruction_template=None):
        if doc["answer"] is not None:
            text = doc["question"]
        else:
            text = self.QUESTION + " " + doc["question"]

        if not instruction_template:
            text = text + "\n" + self.ANSWER

        if instruction_template:
            template = InstructionTemplates.get_template(instruction_template)
            if instruction_template == "orca":
                text = template.format(
                    system_message=self.ORCA_SYSTEM,
                    user_message=text)
            elif instruction_template == 'metamath':
                text = template.format(
                    user_message=text)
            elif instruction_template == 'mathoctopus':
                text = template.format(
                    input_lang=self.LANG_NAME,
                    output_lang=self.LANG_NAME,
                    user_message=text)
        return text

    def doc_to_target(self, doc, instruction_template=None):
        if doc["answer"] is not None:
            return " " + doc["answer"][len(self.ANSWER) + 1:] + '[END]'
        else:
            return " " + str(doc["answer_number"]) + '[END]'

    def construct_requests(self, doc, ctx, instruction_template=None):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        if instruction_template:
            completion = rf.greedy_until(
                ctx, {"until": [self.QUESTION, '[END]', '</s>', '<|im_end|>']})
        else:
            completion = rf.greedy_until(
                ctx, {"until": [self.QUESTION, '[END]']})
        return completion

    def _extract_answer(self, completion):
        # code copied from MathOctopus, the original regex in lm_eval is wrong
        completion = re.sub(r"(\d),(\d)", "\g<1>\g<2>",
                            completion)  # 123,456
        res = re.findall(r"(\d+(\.\d+)?)", completion)  # 123456.789
        if len(res) > 0:
            num_str = res[-1][0]
            return float(num_str)
        else:
            return 0.0

    def _is_correct(self, completion, answer):
        gold = answer
        assert gold != INVALID_ANS, "No ground truth answer found in the document."
        return self._extract_answer(completion) == float(gold)

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        completion = results[0]
        answer = doc["answer_number"]
        return {"acc": self._is_correct(completion, answer)}

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {"acc": mean}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {"acc": True}


class MGSM_English(MGSM):
    DATASET_NAME = "en"
    LANG_NAME = "English"
    QUESTION = "Question:"


class MGSM_Spanish(MGSM):
    DATASET_NAME = "es"
    LANG_NAME = "Spanish"
    QUESTION = "Pregunta:"


class MGSM_French(MGSM):
    DATASET_NAME = "fr"
    LANG_NAME = "French"
    QUESTION = "Question :"


class MGSM_German(MGSM):
    DATASET_NAME = "de"
    LANG_NAME = "German"
    QUESTION = "Frage:"


class MGSM_Russian(MGSM):
    DATASET_NAME = "ru"
    LANG_NAME = "Russian"
    QUESTION = "\u0417\u0430\u0434\u0430\u0447\u0430:"


class MGSM_Chinese(MGSM):
    DATASET_NAME = "zh"
    LANG_NAME = "Chinese"
    QUESTION = "\u95ee\u9898:"


class MGSM_Japanese(MGSM):
    DATASET_NAME = "ja"
    LANG_NAME = "Japanese"
    QUESTION = "\u554f\u984c:"


class MGSM_Thai(MGSM):
    DATASET_NAME = "th"
    LANG_NAME = "Thai"
    QUESTION = "\u0e42\u0e08\u0e17\u0e22\u0e4c:"


class MGSM_Swahili(MGSM):
    DATASET_NAME = "sw"
    LANG_NAME = "Swahili"
    QUESTION = "Swali:"


class MGSM_Bengali(MGSM):
    DATASET_NAME = "bn"
    LANG_NAME = "Bengali"
    QUESTION = "\u09aa\u09cd\u09b0\u09b6\u09cd\u09a8:"


class MGSM_Telugu(MGSM):
    DATASET_NAME = "te"
    LANG_NAME = "Telugu"
    QUESTION = "\u0c2a\u0c4d\u0c30\u0c36\u0c4d\u0c28:"


LANGS = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"]

LANG_CLASSES = [
    MGSM_English,
    MGSM_Spanish,
    MGSM_French,
    MGSM_German,
    MGSM_Russian,
    MGSM_Chinese,
    MGSM_Japanese,
    MGSM_Thai,
    MGSM_Swahili,
    MGSM_Bengali,
    MGSM_Telugu,
]


def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"mgsm_{lang}"] = lang_class
    return tasks
