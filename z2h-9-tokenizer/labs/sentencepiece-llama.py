# Card: sentencepiece-llama.html · Lecture 8 · sentencepiece library intro, used to train Llama 2 vocabulary · https://www.youtube.com/watch?v=zduSFxRajkE&t=5322s
# Run from the course folder: python labs/sentencepiece-llama.py   (writes toy.txt and tok400.model into labs/data/)
import os
import sentencepiece as spm
from common import DATA

os.makedirs(DATA, exist_ok=True)
toy = os.path.join(DATA, 'toy.txt')
with open(toy, 'w', encoding='utf-8') as f:
    f.write("SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end system that does not depend on language-specific pre/postprocessing.")

def train(prefix, byte_fallback):
    # the settings here are (best effort) those used for training Llama 2 (Colab cell 39)
    options = dict(
        input=toy, input_format='text', model_prefix=os.path.join(DATA, prefix),
        model_type='bpe', vocab_size=400,
        normalization_rule_name='identity', remove_extra_whitespaces=False,
        input_sentence_size=200000000, max_sentence_length=4192,
        seed_sentencepiece_size=1000000, shuffle_input_sentence=True,
        character_coverage=0.99995, byte_fallback=byte_fallback,
        split_digits=True, split_by_unicode_script=True, split_by_whitespace=True,
        split_by_number=True, max_sentencepiece_length=16,
        add_dummy_prefix=True, allow_whitespace_only_pieces=True,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
        num_threads=os.cpu_count(), minloglevel=2,
    )
    spm.SentencePieceTrainer.train(**options)
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(DATA, prefix + '.model'))
    return sp

sp = train('tok400', byte_fallback=True)
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]
print('vocab size:', len(vocab))
print('specials:', vocab[:3])
print('byte tokens:', vocab[3:5], '...', vocab[257:259])
print('first merges:', vocab[259:269])
print('last pieces (single code points):', vocab[-10:])
ids = sp.encode('hello 안녕하세요')
print(ids)
print([sp.id_to_piece(idx) for idx in ids])
print("byte 0xEC = 236, its id = 236 + 3 =", sp.piece_to_id('<0xEC>'))
print('decode:', repr(sp.decode(ids)))
print("'world' vs ' world':", sp.encode('world', out_type=str), sp.encode(' world', out_type=str))

sp2 = train('tok400_nofallback', byte_fallback=False)
ids = sp2.encode('hello 안녕하세요')
print('byte_fallback=False:', ids, [sp2.id_to_piece(i) for i in ids])
print('vocab size now:', sp2.get_piece_size(), '| decode:', repr(sp2.decode(ids)))
