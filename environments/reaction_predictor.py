from onmt.translate.silent_translator import build_default_translator
import re

def smi_to_token(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def token_to_smi(token):
    return token.replace(' ', '')

class ReactionPredicter(object):
    def __init__(self):
        self.translator = build_default_translator()
        
    def get_products(self, reactant_lists):
        tokens = [smi_to_token(".".join(reactant_list)) for reactant_list in reactant_lists]
        scores, predictions = self.translator.translate(src_data_iter=tokens, batch_size=32)
        predictions = [token_to_smi(ps[0]) for ps in predictions]
        return scores, predictions

if __name__ == "__main__":
    predicter = ReactionPredicter()
    smis = "C1CCOC1.N#Cc1ccsc1N.O=[N+]([O-])c1cc(F)c(F)cc1F.[H-].[Na+]".split(".")

    import random
    random.shuffle(smis)
    print(smis)
    scores, predictions = predicter.predict([smis])
    print(predictions)