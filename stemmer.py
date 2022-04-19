#Ref: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/#wordnetlemmatizerwithappropriatepostag

import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 



STWDS_BR = stopwords.words('english')

def get_wn_pos(tagged_word):
    
    pos_tag = tagged_word[1].upper()
    """tag_dict = {"JJ": wordnet.ADJ,
                "NN": wordnet.NOUN,
                "VBD": wordnet.VERB,
                "VBN": wordnet.VERB,
                "VBP": wordnet.VERB,
                "RB": wordnet.ADV}
    """
    tag_dict = {"JJ": 'a',
                "NN": 'n',
                "VBD": 'v',
                "VBN": 'v',
                "VBP": 'v',
                "RB": 'r'
                }
                
    return tag_dict.get(pos_tag,'n')


def init_stemming(partition):
    
    lemmatizer = WordNetLemmatizer()
    new_partition = map(lambda r : [r['id'], stemming(str(r['text']), lemmatizer), r['category']], partition)
    return new_partition
    
def stemming(text, lemmatizer)    :
    mos = [s for s in re.finditer("[\.,!,\?]", text)]
    sents = list([])
    prev_index =0
    for mo in mos:
        index = mo.span(0)[0]
        if len(sents) == 0:
            sents = [text[prev_index:index].strip()]
        else:
            sents.append(text[prev_index:index].strip())
        prev_index = mo.span(0)[1]
    #tokenizer, also filters stopwords
    sent_tokens = [[ re.sub('\W+$',"",re.sub('^\W+',"", word.lower())) for word in re.split("[\s+,'\\n'+,\W+]",sent) if (word != "") & (word.lower() not in STWDS_BR)] for sent in sents ] 
   
    lemmatized_text = list([])
    # instead of the broadcasted lemmatizer, you may use lemmatizer = WordNetLemmatizer()
    #though only one is needed per partition not for any row.
    #lemmatizer = LEMMER_BR.value
    
    for sent in sent_tokens:
        pos_tagged_sent = nltk.pos_tag(sent)
        #lemmatized_sent = list([lemmatizer.lemmatize(tagged_word[0], get_wn_pos(tagged_word)) for tagged_word in pos_tagged_sent])
        def lemma(lemmatizer, tagged_word):
            return lemmatizer.lemmatize(tagged_word[0], get_wn_pos(tagged_word))
        lemmatized_sent = map(lambda tagged_word : lemma(lemmatizer, tagged_word), pos_tagged_sent)
        lemmatized_text.extend(lemmatized_sent)
        
    return lemmatized_text    