# fitting-setfits
Playing with SetFit approach for few-shot transfer for text classification.

## Methods Implemented

- **case 0**: The SetFit method as outlined in their paper i.e., a sentence transformer, fine-tuned in a self-supervised
    contrastive manner. Then we slap a logistic classifier on top of encoded sentences and do the actual task.
- **case 1**: This is regular task-specific fine-tuning i.e. over the sentence transformer we
  1. don't do self-supervised fine-tuning of the transformer and directly train for the task 
  2. instead of a logistic classifier we use a regular densenet and train it alongside the encoder
- **case 2**: Similar to case 0 but we don't do self-supervised fine-tuning of the transformer and directly move on 
  to encoding text and fitting a logistic classifier.
- **case 3**: Instead of all this we formulate a few-shot prompt and ask a model on huggingface to classify the text.

## Usage

```commandline
priyansh@priyansh-ele:~/Dev/projects/setfit$ python main.py --help
Usage: main.py [OPTIONS]

Options:
  -d, --dataset-name TEXT         The name of the dataset as it appears on the
                                  HuggingFace hub e.g. SetFit/SentEval-CR |
                                  SetFit/bbc-news | SetFit/enron_spam ...

  -c, --case INTEGER              0, 1, 2, or 3: which experiment are we
                                  running. See readme or docstrings to know
                                  more but briefly: **0**: SentTF ->
                                  Constrastive Pretrain -> +LogReg on task.
                                  **1**: SentTF -> +Dense on task. **2**:
                                  SentTF -> +LogReg on task. **3**:
                                  FewShotPrompting based Clf over Flan-t5-xl
                                  [required]

  -r, --repeat INTEGER            The number of times we should run the entire
                                  experiment (changing the seed).

  -bs, --batch-size INTEGER       ... you know what it is.
  -ns, --num-sents INTEGER        Size of our train set. Set short values
                                  (under 100)

  -e, --num-epochs INTEGER        Epochs for fitting Clf+SentTF on the main
                                  (classification) task.

  -eft, --num-epochs-finetune INTEGER
                                  Epochs for both contrastive pretraining of
                                  SentTF.

  -tot, --test-on-test            If true, we report metrics on testset. If
                                  not, on a 20% split of train set. Off by
                                  default.

  -ft, --full-test                We truncate the testset of every dataset to
                                  be upto 100 instances. If you know what
                                  you're doing, you can test on the full
                                  dataset.NOTE that if you're running this in
                                  case 3 you should probably be a premium
                                  member and not be paying per use.

  --help                          Show this message and exit.

```

> **NOTE**: If you want to query LLMs hosted at huggingface (case 3), you have to 
> create your account on [HuggingFace hub and generate access tokens](https://huggingface.co/settings/tokens)
> after which you should paste them in a file `./hf_token.key`. 
> 
> PS: don't worry I've added this file to .gitignore

Or you can simply run `./run.sh` after installing the required libraries (see `requirements.txt`)

Afterwards, you can run the notebook `summarise.ipynb` to summarize and visualize (if I get around to adding this code)
the results.

## Datasets Used

- [SetFit/SentEval-CR](https://huggingface.co/datasets/SetFit/SentEval-CR)
- [SetFit/bbc-news](https://huggingface.co/datasets/SetFit/bbc-news)
- [SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam/tree/main)

They're all classification datasets that have been cleaned by the nice and kind folks who made the SetFit lib.
But you can use any HF dataset **provided it has these three fields:** 
(i) text (str), (ii) label (int), and (iii) label_text (str).

## Conclusions?

TODO ^^
