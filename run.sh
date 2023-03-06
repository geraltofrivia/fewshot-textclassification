python main.py -c 0 -e 1 -eft 5 -r 5 -tot
python main.py -c 2 -e 1 -eft 5 -r 5 -tot
python main.py -c 1 -e 1 -eft 5 -r 5 -tot
python main.py -c 0 -e 1 -eft 5 -r 5 -tot -ns 10 -bs 5
python main.py -c 2 -e 1 -eft 5 -r 5 -tot -ns 10 -bs 5
python main.py -c 1 -e 1 -eft 5 -r 5 -tot -ns 10 -bs 5
# case 3 is prompting a model so we're gonna stick with ten instances only, not bother with 64
python main.py -c 3 -e 1 -eft 5 -r 1 -tot -ns 10 -bs 5


python main.py -c 0 -e 1 -eft 5 -r 5 -tot -d SetFit/bbc-news
python main.py -c 2 -e 1 -eft 5 -r 5 -tot -d SetFit/bbc-news
python main.py -c 1 -e 1 -eft 5 -r 5 -tot -d SetFit/bbc-news
python main.py -c 0 -e 1 -eft 5 -r 5 -tot -ns 10 -bs 5 -d SetFit/bbc-news
python main.py -c 2 -e 1 -eft 5 -r 5 -tot -ns 10 -bs 5 -d SetFit/bbc-news
python main.py -c 1 -e 1 -eft 5 -r 5 -tot -ns 10 -bs 5 -d SetFit/bbc-news
# case 3 is prompting a model so we're gonna stick with ten instances only, not bother with 64
python main.py -c 3 -e 1 -eft 5 -r 1 -tot -ns 10 -bs 5 -d SetFit/bbc-news


python main.py -c 0 -e 1 -eft 5 -r 5 -tot -d SetFit/enron_spam
python main.py -c 2 -e 1 -eft 5 -r 5 -tot -d SetFit/enron_spam
python main.py -c 1 -e 1 -eft 5 -r 5 -tot -d SetFit/enron_spam
python main.py -c 0 -e 1 -eft 5 -r 5 -tot -ns 10 -bs 5 -d SetFit/enron_spam
python main.py -c 2 -e 1 -eft 5 -r 5 -tot -ns 10 -bs 5 -d SetFit/enron_spam
python main.py -c 1 -e 1 -eft 5 -r 5 -tot -ns 10 -bs 5 -d SetFit/enron_spam
# case 3 is prompting a model so we're gonna stick with ten instances only, not bother with 64
python main.py -c 3 -e 1 -eft 5 -r 1 -tot -ns 10 -bs 5 -d SetFit/enron_spam