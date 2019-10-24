from stanford_nlp import StanfordNLP

class StanfordPipeline:
    def __init__(self, processors='tokenize,pos,lemma'):
        self.sf_nlp = StanfordNLP()
        self.pipeline = self.sf_nlp.Pipeline(processors = processors)

    def extract_named_entities(doc):
        parsed_text = {'word':[], 'type':[]}
        for sent in doc.sentences:
            for wrd in sent.words:
                #extract text and lemma
                parsed_text['word'].append(wrd.text)
                parsed_text['ne'].append(wrd.lemma)
        #return a dataframe
        return pd.DataFrame(parsed_text)
