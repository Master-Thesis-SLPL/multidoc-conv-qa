from __future__ import absolute_import, division, print_function

import json
import logging
import os

import datasets

MAX_Q_LEN = 100  # Max length of question
YOUR_LOCAL_DOWNLOAD = "../../dataset"

_CITATION = """\
    @inproceedings{choi2018quac,
    title={QuAC: Question Answering in Context},
    author={Choi, Eunsol and He, He and Iyyer, Mohit and Yatskar, Mark and Yih, Wen-tau and Choi, Yejin and Liang, Percy and Zettlemoyer, Luke},
    booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
    pages={2174--2184},
    year={2018}
    }
"""

_DESCRIPTION = """\
    QuAC
"""

_HOMEPAGE = "https://quac.ai/"


_URLs = "https://s3.amazonaws.com/my89public/quac/train_v0.2.json, https://s3.amazonaws.com/my89public/quac/val_v0.2.json"

class Quac(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.2.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="quac_rc",
            version=VERSION,
            description="Load QuAC dataset for machine reading comprehension tasks",
        )
    ]

    DEFAULT_CONFIG_NAME = "quac_rc"

    def _info(self):

        if self.config.name == "quac_rc":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                            # "spans": datasets.features.Sequence(datasets.Value("string"))
                        }
                    ),
                    "domain": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager):

        my_urls = _URLs

        # data_dir = dl_manager.download_and_extract(my_urls) 
        data_dir = YOUR_LOCAL_DOWNLOAD # point to local dir to avoid downloading the dataset again

        if self.config.name == "quac_rc":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "quac/val_v0.2.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "quac/train_v0.2.json"
                        ),
                    },
                ),
            ]
    
    def _get_answers(self, answers):
        selected_answers = []
        for answer in answers:
            if len(answer["text"]) > 0 and answer["text"] != "CANNOTANSWER":
                selected_answer = {"text": answer["text"], "answer_start": answer["answer_start"]}
                selected_answers.append(selected_answer)

        if len(selected_answers) > 0:
            return selected_answers 
        else:
            return None

    def _generate_examples(self, filepath):
        if self.config.name == "quac_rc":
            """Load dialog data in the reading comprehension task setup, where context is the grounding document,
            input query is dialog history in reversed order, and output to predict is the next agent turn."""

            logging.info("generating examples from = %s", filepath)
            with open(filepath, "r") as f:
                data = json.load(f)["data"]

            for group in data:
                domain = group["title"]
                for item in group['paragraphs']:
                    title = item["id"]
                    context = item["context"]
                    all_prev_utterances = []
                    for qa in item["qas"]:
                        id_ = qa["id"]
                        all_prev_utterances.append(qa["question"])
                        answer = self._get_answers(qa["answers"]) # could be modified

                        if answer is not None:
                            question_str = " ".join(
                                        list(reversed(all_prev_utterances))
                                    ).strip()
                            question = " ".join(question_str.split()[:MAX_Q_LEN])

                            # append the original answer into the utterance list
                            orig_answer_text = qa["orig_answer"]["text"]
                            all_prev_utterances.append(orig_answer_text)

                            qa = {
                                    "id": id_, # For subtask1, the id should be this format.
                                    "title": title,
                                    "context": context,
                                    "question": question,
                                    "answers": answer,  # For subtask1, "answers" contains the grounding annotations for evaluation.
                                    "domain": domain,
                                }
                            yield id_, qa
                            