from pathlib import Path

import numpy as np
from langchain import HuggingFaceHub, PromptTemplate, FewShotPromptTemplate, LLMChain
from langchain.prompts.example_selector import LengthBasedExampleSelector
from tqdm.auto import tqdm
from transformers import AutoConfig
from mytorch.utils.goodies import FancyDict


class PromptClassifier:
    """
        This prompts Flan T5 XL to do classification using LangChain.
        Give it train and test dataset and let it RIP
    """

    def __init__(self, dataset, seed, num_sents, test_on_test, *args, **kwargs):

        # First figure out the length of the model
        config = AutoConfig.from_pretrained("google/flan-t5-xl")
        max_len = config['n_positions']

        # # initialize Hub LLM
        hub_llm = HuggingFaceHub(
                repo_id='google/flan-t5-xl',
            model_kwargs={'temperature':1e-10}
        )

        # Read HuggingFace API key
        try:
            with (Path('.') / 'hf_token.key').open('r') as f:
                hf_token_key = f.read().split()
        except FileNotFoundError:
            raise FileNotFoundError(f"No HuggingFace API key found at {(Path('.') / 'hf_token.key').absolute()}"
                                    f"You need to generate yours at https://huggingface.co/settings/tokens"
                                    f"and paste it in this file.")

        # Here we be validatin' if the dataset fits the template or not (have 'text' and 'label_text'
        # TODO: for now pre-check datasets and send here

        # Go through the dataset, generate train and testset
        # Sample num_sents from the dataset. Divide them in 80/20
        train_ds = dataset["train"].shuffle(seed=seed).select(range(num_sents))
        if test_on_test:
            test_ds = dataset["test"]
        else:
            train_ds, test_ds = train_ds.select(
                range(int(len(train_ds) * 0.8))
            ), train_ds.select(range(int(len(train_ds) * 0.8), len(train_ds)))


        """ Prompt stuff """
        # create a example template
        example_template = """
        Review: {query}
        Sentiment: {answer}
        """
        # create a prompt example from above template
        example_prompt = PromptTemplate(
            input_variables=["query", "answer"],
            template=example_template
        )
        examples = [{"query": x['text'], "answer": x["label_text"]} for x in train_ds]
        prefix = """Classify into positive or negative. Here are some examples: """
        suffix = """
        Review: {query}
        Sentiment: 
        """
        # We'll use the `LengthBasedExampleSelector` to select the examples.
        example_selector = LengthBasedExampleSelector(
            # These are the examples is has available to choose from.
            examples=examples,
            # This is the PromptTemplate being used to format the examples.
            example_prompt=example_prompt,
            # This is the maximum length that the formatted examples should be.
            # Length is measured by the get_text_length function below.
            max_length=max_len,
        )

        # now create the few shot prompt template
        few_shot_prompt_template = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["query"],
            example_separator="\n\n"
        )
        self.llm_chain = LLMChain(
            prompt=few_shot_prompt_template,
            llm=hub_llm
        )

        self.train_ds = train_ds
        self.test_ds = test_ds

    def run(self):
        score = []

        for i, item in enumerate(tqdm(self.test_ds)):
            answer = self.llm_chain.run(item['text'])

            if answer.strip().lower() == item['label_text'].strip().lower():
                score.append(1)
                continue

            if len(item['label_text'])*0.5 > len(answer) > len(item['label_text'])*2:
                raise ValueError(f"The answer to {i}th element is `{answer}`. \n"
                                 f"Expected `{item['label_text']}`. ")

            score.append(0)
        return {"accuracy": np.mean(score)}


if __name__ == '__main__':
    config = FancyDict(
        **{
            "num_sents": 8,
            "test_on_test": False,
        }
    )

    ds = load_dataset("SetFit/SentEval-CR")