#find out why this does not work, it shoud workd because sacreBLEU rocks

from sacrebleu.metrics import BLEU, CHRF, TER
 
refs = [ ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'], 
['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.'],]

sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

bleu = BLEU()

print(bleu.corpus_score(sys, refs))
print(bleu.get_signature())


chrf = CHRF()

print(chrf.corpus_score(sys, refs))

