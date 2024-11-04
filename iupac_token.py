import re
import pandas as pd
import os
from typing import List, Optional
import random


def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def split_tokens(tokens_to_be_splitted, all_tokens):
    if ''.join(tokens_to_be_splitted) in all_tokens:
        indexes = duplicates(all_tokens, ''.join(tokens_to_be_splitted))
        for k, idx in enumerate(indexes):
            all_tokens[idx] = tokens_to_be_splitted[0]
            for i in range(1, len(tokens_to_be_splitted)):
                all_tokens.insert(idx+i, tokens_to_be_splitted[i])
            if k < len(indexes) - 1:
                indexes[k+1:] = list(map(lambda x: x+(len(tokens_to_be_splitted) - 1), indexes[k+1:]))
def add_symbol(pat):
    new_patt = pat[1:].split(',')
    ls_del = []
    if new_patt[0][0] == '0':
        ls_del.append(new_patt[0][0])
        new_patt[0] = new_patt[0][1:]

    while int(new_patt[0]) > int(new_patt[1]):
        ls_del.append(new_patt[0][0])
        new_patt[0] = new_patt[0][1:]
    result = ['.', ''.join(ls_del), '^', new_patt[0], ',', new_patt[1]]
    return result

def check_correct_tokenize(iupac,tokens):
    if ''.join(tokens).replace('^', '') == iupac:
        return True
    else:
        return False

def iupac_tokenizer(iupac, is_check=True, construct_dict=False, token_json=None):

    pattern = "\.\d+,\d+|[1-9]{2}\-[a-z]\]|[0-9]\-[a-z]\]|[1-9]{2}[a-z]|[1-9]{2}'[a-z]|[0-9]'[a-z]|[0-9][a-z]|\([0-9]\+\)|\([0-9]\-\)|" + \
              "[1-9]{2}|[0-9]|-|\s|\(|\)|S|R|E|Z|N|C|O|'|\"|;|λ|H|,|\.|\[[a-z]{2}\]|\[[a-z]\]|\[|\]|"

    alcane = "methane|methanoyl|methan|ethane|ethanoyl|ethan|propanoyl|propane|propan|propa|butane|butanoyl|butan|buta|pentane|" + \
             "pentanoyl|pentan|hexane|hexanoyl|hexan|heptane|heptanoyl|heptan|octane|octanoyl|octan|nonane|nonanoyl|" + \
             "nonan|decane|decanoyl|decan|icosane|icosan|cosane|cosan|contane|contan|"

    pristavka_name = "hydroxide|hydroxyl|hydroxy|hydrate|hydro|cyclo|spiro|iso|"
    pristavka_digit = "mono|un|bis|bi|dicta|di|tetraza|tetraz|tetra|tetr|pentaza|pentaz|penta|hexaza|" + \
                      "hexa|heptaza|hepta|octaza|octa|nonaza|nona|decaza|deca|kis|"

    prefix_alcane = "methylidene|methyl|ethyl|isopropyl|propyl|isobutyl|sec-butyl|tert-butyl|butyl|pentyl|hexyl|heptyl|octyl|"
    carbon = "meth|eth|prop|but|pent|hex|hept|oct|non|dec|icosa|icos|cosa|cos|icon|conta|cont|con|heni|hene|hen|hecta|hect|"

    prefix_all = "benzhydryl|benzoxaza|benzoxaz|benzoxa|benzox|benzo|benzyl|benz|phenacyl|phenanthro|phenyl|phenoxaza|phenoxaz|phenoxy|phenox|phenol|pheno|phen|acetyl|aceto|acet|" + \
                 "peroxy|oxido|oxino|oxalo|oxolo|oxocyclo|oxol|oxoc|oxon|oxo|oxy|pyrido|pyrimido|imidazo|naphtho|stiboryl|stibolo|"

    prefix_isotope = "protio|deuterio|tritio|"
    suffix_isotope = "protide|"
    prefix_galogen = "fluoro|fluoranyl|fluoridoyl|fluorido|chloro|chloranyl|chloridoyl|chlorido|bromo|bromanyl|bromidoyl|bromido|iodo|iodanyl|iodidoyl|iodanuidyl|iodido|"
    suffix_galogen = "fluoride|chloride|chloridic|perchloric|bromide|iodide|iodane|hypoiodous|hypochlorous|"
    prefix_сhalcogen = "phosphonato|phosphoroso|phosphonia|phosphoryl|phosphanyl|arsono|arsanyl|stiba|"
    suffix_сhalcogen = "phosphanium|phosphate|phosphite|phosphane|phosphanide|phosphonamidic|phosphonous|phosphinous|phosphinite|phosphono|arsonic|stibane|"
    prefix_metal = "alumanyl|gallanyl|stannyl|plumbyl|"
    suffix_metal = "chromium|stannane|gallane|alumane|aluminane|aluminan|"
    prefix_non_metal = "tellanyl|germanyl|germyl|"
    suffix_non_metal = "germane|germa|"

    prefix_sulfur = "sulfanylidene|sulfinamoyl|sulfonimidoyl|sulfinimidoyl|sulfamoyl|sulfonyl|sulfanyl|sulfinyl|sulfinato|sulfenato|" + \
                    "sulfonato|sulfonio|sulfino|sulfono|sulfido|"
    suffix_sulfur = "sulfonamide|sulfinamide|sulfonamido|sulfonic|sulfamic|sulfinic|sulfuric|thial|thione|thiol|" + \
                    "sulfonate|sulfite|sulfate|sulfide|sulfinate|sulfanium|sulfamate|sulfane|sulfo|"

    prefix_nitrogen = "hydrazono|hydrazino|nitroso|nitrous|nitro|formamido|amino|amido|imino|imido|anilino|anilin|thiocyanato|cyanato|cyano|azido|azanidyl|azanyl|" + \
                      "azanide|azanida|azonia|azonio|amidino|nitramido|diazo|"
    suffix_nitrogen = "ammonium|hydrazide|hydrazine|hydrazin|amine|imine|oxamide|nitramide|formamide|cyanamide|amide|imide|amidine|isocyanide|azanium|" + \
                      "thiocyanate|cyanate|cyanic|cyanatidoyl|cyanide|nitrile|nitrite|hydrazonate|"

    suffix_carbon = "carbonitrile|carboxamide|carbamimidothioate|carbodithioate|carbohydrazonate|carbonimidoyl|carboximidoyl|" + \
                    "carbamimidoyl|carbamimidate|carbamimid|carbaldehyde|carbamate|carbothioyl|carboximidothioate|carbonate|" + \
                    "carboximidamide|carboximidate|carbamic|carbonochloridate|carbothialdehyde|carbothioate|carbothioic|carbono|carbon|carbo|" + \
                    "formate|formic|"
    prefix_carbon = "carboxylate|carboxylato|carboxylic|carboxy|halocarbonyl|carbamoyl|carbonyl|carbamo|thioformyl|formyl|"

    silicon = "silanide|silane|silole|silanyl|silyloxy|silylo|silyl|sila|"
    boron = "boranyl|boranuide|boronamidic|boranuida|boranide|borinic|borate|borane|boran|borono|boron|bora|"
    selenium = "selanyl|seleno|"

    suffix_all = "ane|ano|an|ene|enoxy|eno|en|yne|yn|yl|peroxol|peroxo|" + \
                 "terephthalate|terephthalic|phthalic|phthalate|oxide|oate|ol|oic|ic|al|ate|ium|one|"

    carbon_trivial = "naphthalen|naphthal|inden|adamant|fluoren|thiourea|urea|anthracen|acenaphthylen|" + \
                     "carbohydrazide|annulen|aniline|acetaldehyde|benzaldehyde|formaldehyde|phthalaldehyde|acephenanthrylen|" + \
                     "phenanthren|chrysen|carbanid|chloroform|fulleren|cumen|formonitril|fluoranthen|terephthalaldehyde|azulen|picen|" + \
                     "pyren|pleiaden|coronen|tetracen|pentacen|perylen|pentalen|heptalen|cuban|hexacen|oxanthren|ovalen|aceanthrylen|"

    heterocycles = "indolizin|arsindol|indol|furan|furo|piperazin|pyrrolidin|pyrrolizin|thiophen|thiolo|imidazolidin|imidazol|pyrimidin|pyridin|" + \
                    "piperidin|morpholin|pyrazol|pyridazin|oxocinnolin|cinnolin|pyrrol|thiochromen|oxochromen|chromen|quinazolin|phthalazin|quinoxalin|carbazol|xanthen|pyrazin|purin|" + \
                    "indazol|naphthyridin|quinolizin|guanidin|pyranthren|pyran|thianthren|thian|acridin|acrido|yohimban|porphyrin|pteridin|tetramin|pentamin|" + \
                    "borinin|borino|boriran|borolan|borol|borinan|phenanthridin|quinolin|perimidin|corrin|phenanthrolin|phosphinolin|indacen|silonin|borepin|"

    prefix_heterocycles = "thiaz|oxaza|oxaz|oxan|oxa|ox|aza|az|thia|thioc|thion|thio|thi|telluro|phospha|phosph|selen|bor|sil|alum|ars|germ|tellur|imid|"

    suffix_heterocycles = "ir|et|olo|ol|ino|in|ep|oc|on|ec|"
    saturated_unsatured = "idine|idene|idin|ane|an|ine|in|id|e|"
    pristavka_exception = "do|trisodium|tris|triacetyl|triamine|triaza|triaz|tria|trityl|tri|o"

    type_ = "acid|ether|"
    element = "hydrogen|helium|lithium|beryllium|nitrogen|oxygen|fluorine|neon|sodium|magnesium|aluminum|silicon|" + \
              "phosphorus|sulfur|chlorine|argon|potassium|calcium|scandium|titanium|vanadium|chromium|manganese|iron|" + \
              "cobalt|nickel|copper|zinc|gallium|germanium|arsenic|selenium|bromine|krypton|rubidium|yttrium|zirconium|" + \
              "niobium|molybdenum|technetium|ruthenium|rhodium|palladium|silver|cadmium|indium|antimony|tellurium|iodine|" + \
              "xenon|cesium|barium|lanthanum|cerium|praseodymium|neodymium|latinum|promethium|samarium|europium|gadolinium|" + \
              "terbium|dysprosium|holmium|erbium|thulium|ytterbium|lutetium|hafnium|tantalum|tungsten|rhenium|osmium|" + \
              "iridium|platinum|gold|aurum|mercury|thallium|lead|bismuth|polonium|astatine|radon|francium|radium|actinium|" + \
              "thorium|protactinium|uranium|neptunium|plutonium|americium|curium|berkelium|einsteinium|fermium|californium|" + \
              "mendelevium|nobelium|lawrencium|rutherfordium|dubnium|seaborgium|bohrium|hassium|meitnerium|tin|"

    other_ions = "perchlorate|perbromate|periodate|hypofluorite|hypochlorite|hypobromite|hypoiodite|nitrate|silicate|hydride|"

    if construct_dict:
      type_lst = ['alcane', 'pristavka_name', 'pristavka_digit', 'prefix_alcane', 'carbon', 'prefix_all',\
      'prefix_isotope', 'prefix_galogen', 'suffix_galogen', 'prefix_сhalcogen', 'suffix_сhalcogen', \
      'prefix_metal', 'suffix_metal', 'prefix_non_metal', 'suffix_non_metal', 'prefix_sulfur', 'suffix_sulfur', \
      'prefix_nitrogen', 'suffix_nitrogen', 'suffix_carbon', 'prefix_carbon', 'silicon', 'boron','selenium', \
      'suffix_all', 'carbon_trivial', 'heterocycles', 'prefix_heterocycles', 'suffix_heterocycles', 'saturated_unsatured', \
      'pristavka_exception', 'type_', 'element', 'other_ions']
      # erase suffix_isotope, because this class has only one sample. 
      type_dict = {}
      for tp in type_lst:
        res_set = set(locals()[tp].split("|"))
        if '' in res_set:
            res_set.remove('')
        
        # filter the element not appear in tokens
        res_set_filter = set()
        for ele in list(res_set):
          if ele in token_json:
            res_set_filter.add(ele)    
        
        type_dict[tp] = res_set_filter
  
      
      revert_dict = {}
      for tp, values in type_dict.items():
          for va in values:
              revert_dict[va] = tp
      return type_dict, revert_dict      



    regex = re.compile(pattern + heterocycles + carbon_trivial + type_ + element + prefix_isotope + other_ions + alcane + pristavka_digit + pristavka_name + prefix_alcane + \
                       carbon + silicon + prefix_nitrogen + prefix_sulfur + prefix_carbon + prefix_metal + prefix_non_metal + prefix_all + prefix_galogen + prefix_сhalcogen + \
                       suffix_carbon + suffix_nitrogen + suffix_sulfur + suffix_galogen + suffix_сhalcogen + suffix_metal + suffix_non_metal + suffix_all + suffix_heterocycles + \
                       suffix_isotope + boron + selenium  + prefix_heterocycles + saturated_unsatured + pristavka_exception)
    tokens = [token for token in regex.findall(iupac)]

    split_tokens(['meth', 'ane'], tokens)
    split_tokens(['meth', 'an'], tokens)
    split_tokens(['eth', 'ane'], tokens)
    split_tokens(['eth', 'an'], tokens)
    split_tokens(['prop', 'ane'], tokens)
    split_tokens(['prop', 'an'], tokens)
    split_tokens(['but', 'ane'], tokens)
    split_tokens(['but', 'an'], tokens)
    split_tokens(['pent', 'ane'], tokens)
    split_tokens(['pent', 'an'], tokens)
    split_tokens(['hex', 'ane'], tokens)
    split_tokens(['hex', 'an'], tokens)
    split_tokens(['hept', 'ane'], tokens)
    split_tokens(['hept', 'an'], tokens)
    split_tokens(['oct', 'ane'], tokens)
    split_tokens(['oct', 'an'], tokens)
    split_tokens(['non', 'ane'], tokens)
    split_tokens(['non', 'an'], tokens)
    split_tokens(['dec', 'ane'], tokens)
    split_tokens(['dec', 'an'], tokens)
    split_tokens(['cos', 'ane'], tokens)
    split_tokens(['cos', 'an'], tokens)
    split_tokens(['cont', 'ane'], tokens)
    split_tokens(['cont', 'an'], tokens)
    split_tokens(['icos', 'ane'], tokens)
    split_tokens(['icos', 'an'], tokens)

    split_tokens(['thi', 'az'], tokens)
    split_tokens(['thi', 'oc'], tokens)
    split_tokens(['thi', 'on'], tokens)
    split_tokens(['benz', 'ox'], tokens)
    split_tokens(['benz', 'oxa'], tokens)
    split_tokens(['benz', 'ox', 'az'], tokens)
    split_tokens(['benz', 'ox', 'aza'], tokens)
    split_tokens(['phen', 'ox'], tokens)
    split_tokens(['phen', 'oxy'], tokens)
    split_tokens(['phen', 'oxa'], tokens)
    split_tokens(['phen', 'ox', 'az'], tokens)
    split_tokens(['phen', 'ox', 'aza'], tokens)
    split_tokens(['phen', 'ol'], tokens)
    split_tokens(['en', 'oxy'], tokens)
    split_tokens(['ox', 'az'], tokens)
    split_tokens(['ox', 'aza'], tokens)
    split_tokens(['tri', 'az'], tokens)
    split_tokens(['tri', 'amine'], tokens)
    split_tokens(['tri', 'acetyl'], tokens)
    split_tokens(['ox', 'ol'], tokens)
    split_tokens(['ox', 'olo'], tokens)
    split_tokens(['ox', 'an'], tokens)
    split_tokens(['ox', 'oc'], tokens)
    split_tokens(['ox', 'on'], tokens)
    split_tokens(['tri', 'az'], tokens)
    split_tokens(['tri', 'aza'], tokens)
    split_tokens(['tri', 'sodium'], tokens)
    split_tokens(['tetr', 'az'], tokens)
    split_tokens(['tetr', 'aza'], tokens)
    split_tokens(['pent', 'az'], tokens)
    split_tokens(['pent', 'aza'], tokens)
    split_tokens(['hex', 'aza'], tokens)
    split_tokens(['hept', 'aza'], tokens)
    split_tokens(['oct', 'aza'], tokens)
    split_tokens(['non', 'aza'], tokens)
    split_tokens(['dec', 'aza'], tokens)
    split_tokens(['oxo', 'chromen'], tokens)
    split_tokens(['oxo', 'cinnolin'], tokens)
    split_tokens(['oxo', 'cyclo'], tokens)
    split_tokens(['thio', 'chromen'], tokens)
    split_tokens(['thio', 'cyanato'], tokens)

    if (len(re.findall(re.compile('[0-9]{2}\-[a-z]\]'), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile('[0-9]{2}\-[a-z]\]'), tok):
                tokens[i] = tok[:2]
                tokens.insert(i+1,tok[2])
                tokens.insert(i+2,tok[3])
                tokens.insert(i+3,tok[4])

    if (len(re.findall(re.compile('[0-9]\-[a-z]\]'), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile('[0-9]\-[a-z]\]'), tok):
                tokens[i] = tok[:1]
                tokens.insert(i+1,tok[1])
                tokens.insert(i+2,tok[2])
                tokens.insert(i+3,tok[3])

    if (len(re.findall(re.compile('\[[a-z]{2}\]'), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile('\[[a-z]{2}\]'), tok):
                tokens[i] = tok[0]
                tokens.insert(i+1,tok[1])
                tokens.insert(i+2,tok[2])
                tokens.insert(i+3,tok[3])

    if (len(re.findall(re.compile('\[[a-z]\]'), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile('\[[a-z]\]'), tok):
                tokens[i] = tok[0]
                tokens.insert(i+1,tok[1])
                tokens.insert(i+2,tok[2])

    if (len(re.findall(re.compile("[0-9]{2}'[a-z]"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("[0-9]{2}'[a-z]"), tok):
                tokens[i] = tok[:2]
                tokens.insert(i+1,tok[2])
                tokens.insert(i+2,tok[3])

    if (len(re.findall(re.compile("[0-9]'[a-z]"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("[0-9]'[a-z]"), tok):
                tokens[i] = tok[0]
                tokens.insert(i+1,tok[1])
                tokens.insert(i+2,tok[2])

    if (len(re.findall(re.compile("[0-9]{2}[a-z]"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("[0-9]{2}[a-z]"), tok):
                tokens[i] = tok[:2]
                tokens.insert(i+1,tok[2])

    if (len(re.findall(re.compile("[0-9][a-z]"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("[0-9][a-z]"), tok):
                tokens[i] = tok[0]
                tokens.insert(i+1,tok[1])

    if (len(re.findall(re.compile("\.\d+,\d+"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("\.\d+,\d+"), tok):
                result = add_symbol(tok)
                tokens[i] = result[0]
                for k, res in enumerate(result[1:], start=1):
                    tokens.insert(i+k,res)

    if check_correct_tokenize(iupac, tokens) == True:
        return tokens
    else:
        if not is_check:
            return tokens
        else:
            return None

def smiles_tokenizer(smi):
  pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
  regex = re.compile(pattern)
  tokens = [token for token in regex.findall(smi)]
  return tokens

################################################################

from tqdm import tqdm
import torch.multiprocessing as mp
import time

def map_parallel (lst, fn, nworkers=1, fallback_sharing=False):
    if nworkers == 1:
        return [fn(item) for item in lst]

    if fallback_sharing:
        mp.set_sharing_strategy('file_system')
    L = mp.Lock()
    QO = mp.Queue(nworkers)
    QI = mp.Queue()
    for item in lst:
        QI.put(item)

    def pfn ():
        time.sleep(0.0001)
        #print(QI.empty(), QI.qsize())
        while QI.qsize() > 0: #not QI.empty():
            L.acquire()
            item = QI.get()
            L.release()
            obj = fn(item)
            QO.put(obj)

    procs = []
    for nw in range(nworkers):
        P = mp.Process(target=pfn, daemon=True)
        time.sleep(0.0001)
        P.start()
        procs.append(P)

    return [QO.get() for i in range(len(lst))]


def reverse_vocab (vocab):
    return dict((v,k) for k,v in vocab.items())

class IupacModel ():
    def __init__ (self, names, nworkers=1):
        #self.special_tokens =  ['<pad>', '<unk>', '<bos>', '<eos>', ';', '.', '>']
        self.special_tokens =  ["<s>","<pad>","</s>","<unk>","<mask>", ';', '.', '>']
        vocab = set()
        names = map_parallel(names, iupac_tokenizer, nworkers)
        for i in tqdm(range(len(names))):
            tokens = names[i]
            if tokens is not None:
                tokens = set(tokens)
                tokens -= set(self.special_tokens)
                vocab |= tokens
        nspec = len(self.special_tokens)

        self.vocab = {}
        for i,spec in enumerate(self.special_tokens):
            self.vocab[spec] = i
        self.vocab.update(dict(zip(sorted(vocab),
                              range(nspec, nspec+len(vocab)))))
        self.rev_vocab = reverse_vocab(self.vocab)
        self.vocsize = len(self.vocab)

        with open('/sharefs/ylx/chem_data/results/tokenizer_dir/iupac_regex/vocab.json','w') as w:
            json.dump(self.vocab,w)

    def encode (self, seq):
        tokens = iupac_tokenizer(seq)
        if tokens is None:
            raise Exception(f"Unable to tokenize IUPAC name: {seq}")
        else:
            return [self.vocab[t] for t in tokens]

    def decode (self, seq):
        seq = [s for s in seq if s > 4]
        return "".join([self.rev_vocab[code] for code in seq])

class SmilesModel ():
  def __init__ (self, names, nworkers=1):
      #self.special_tokens =  ['<pad>', '<unk>', '<bos>', '<eos>', ';', '.', '>']
      self.special_tokens =  ["<s>","<pad>","</s>","<unk>","<mask>", ';', '.', '>']
      vocab = set()
      names = map_parallel(names, smiles_tokenizer, nworkers)
      for i in tqdm(range(len(names))):
          tokens = names[i]
          if tokens is not None:
              tokens = set(tokens)
              tokens -= set(self.special_tokens)
              vocab |= tokens
      nspec = len(self.special_tokens)

      self.vocab = {}
      for i,spec in enumerate(self.special_tokens):
          self.vocab[spec] = i
      self.vocab.update(dict(zip(sorted(vocab),
                            range(nspec, nspec+len(vocab)))))
      self.rev_vocab = reverse_vocab(self.vocab)
      self.vocsize = len(self.vocab)

      with open('/sharefs/ylx/chem_data/results/tokenizer_dir/smiles_regex/vocab.json','w') as w:
          json.dump(self.vocab,w)

  def encode (self, seq):
      tokens = smiles_tokenizer(seq)
      if tokens is None:
          raise Exception(f"Unable to tokenize IUPAC name: {seq}")
      else:
          return [self.vocab[t] for t in tokens]

  def decode (self, seq):
      seq = [s for s in seq if s > 4]
      return "".join([self.rev_vocab[code] for code in seq])


import os
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers import BertTokenizer,RobertaTokenizer,RobertaTokenizerFast
import collections
import json
from typing import List

from logging import getLogger

logger = getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}
class IUPACTokenizer(PreTrainedTokenizer):
  """
  Tokenizer in RobertaTokenizer style.
  Creates the SmilesTokenizer class. The tokenizer heavily inherits from the BertTokenizer
  implementation found in Huggingface's transformers library. It runs a WordPiece tokenization
  algorithm over SMILES strings using the tokenisation SMILES regex developed by Schwaller et. al.

  Please see https://github.com/huggingface/transformers
  and https://github.com/rxn4chemistry/rxnfp for more details.

  Examples
  --------
  >>> from deepchem.feat.smiles_tokenizer import SmilesTokenizer
  >>> current_dir = os.path.dirname(os.path.realpath(__file__))
  >>> vocab_path = os.path.join(current_dir, 'tests/data', 'vocab.txt')
  >>> tokenizer = SmilesTokenizer(vocab_path)
  >>> print(tokenizer.encode("CC(=O)OC1=CC=CC=C1C(=O)O"))
  [12, 16, 16, 17, 22, 19, 18, 19, 16, 20, 22, 16, 16, 22, 16, 16, 22, 16, 20, 16, 17, 22, 19, 18, 19, 13]


  References
  ----------
  .. [1] Schwaller, Philippe; Probst, Daniel; Vaucher, Alain C.; Nair, Vishnu H; Kreutter, David;
     Laino, Teodoro; et al. (2019): Mapping the Space of Chemical Reactions using Attention-Based Neural
     Networks. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.9897365.v3

  Note
  ----
  This class requires huggingface's transformers and tokenizers libraries to be installed.
  """
  vocab_files_names = VOCAB_FILES_NAMES

  def __init__(
      self,
      vocab_file: str = '',
      bos_token="<s>",
      eos_token="</s>",
      sep_token="</s>",
      cls_token="<s>",
      unk_token="<unk>",
      pad_token="<pad>",
      mask_token="<mask>",
      add_prefix_space=False,
      **kwargs):
    """Constructs a SmilesTokenizer.

    Parameters
    ----------
    vocab_file: str
      Path to a SMILES character per line vocabulary file.
      Default vocab file is found in deepchem/feat/tests/data/vocab.txt
    """

    bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
    eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
    sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
    cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
    unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
    pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

    # Mask token behave like a normal word, i.e. include the space before it
    mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

    super().__init__(
        vocab_file=vocab_file,
        bos_token=bos_token,
        eos_token=eos_token,
        unk_token=unk_token,
        sep_token=sep_token,
        cls_token=cls_token,
        pad_token=pad_token,
        mask_token=mask_token,
        add_prefix_space=add_prefix_space,
        **kwargs,
    )


    #super().__init__(vocab_file, **kwargs) #merges_file
    # take into account special tokens in max length
    # self.max_len_single_sentence = self.model_max_length - 2
    # self.max_len_sentences_pair = self.model_max_length - 3

    if not os.path.isfile(vocab_file):
      raise ValueError(
          "Can't find a vocab file at path '{}'.".format(vocab_file))
    with open(vocab_file, 'r') as vr:
        self.vocab = json.load(vr)
        # self.vocab = load_vocab(vocab_file)
    # self.highest_unused_index = max(
    #     [i for i, v in enumerate(self.vocab.keys()) if v.startswith("[unused")])
    self.ids_to_tokens = collections.OrderedDict(
        [(ids, tok) for tok, ids in self.vocab.items()])
    
    self.basic_tokenizer = iupac_tokenizer
    
    
    self.type_dict, self.revert_dict = self.basic_tokenizer('', construct_dict=True, token_json=self.vocab)
    
    
    self.init_kwargs["model_max_length"] = self.model_max_length

  @property
  def vocab_size(self):
    return len(self.vocab)

  @property
  def vocab_list(self):
    return list(self.vocab.keys())

  def _tokenize(self, text: str):
    """Tokenize a string into a list of tokens.

    Parameters
    ----------
    text: str
      Input string sequence to be tokenized.
    """

    split_tokens = [token for token in self.basic_tokenizer(text, is_check=False)]
    return split_tokens

  def build_inputs_with_special_tokens(
      self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
      """
      Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
      adding special tokens. A RoBERTa sequence has the following format:

      - single sequence: ``<s> X </s>``
      - pair of sequences: ``<s> A </s></s> B </s>``

      Args:
          token_ids_0 (:obj:`List[int]`):
              List of IDs to which the special tokens will be added.
          token_ids_1 (:obj:`List[int]`, `optional`):
              Optional second list of IDs for sequence pairs.

      Returns:
          :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
      """
      if token_ids_1 is None:
          return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
      cls = [self.cls_token_id]
      sep = [self.sep_token_id]
      return cls + token_ids_0 + sep + sep + token_ids_1 + sep
  def tokenize_replace(self, text: str, max_seq_length=128):
    split_tokens = self._tokenize(text)
    random_candi_lst = []
    for idx, tk in enumerate(split_tokens):
      if tk not in self.revert_dict:
        continue
      class_k = self.revert_dict[tk]
      candidate_set = self.type_dict[class_k].copy()
      candidate_set.remove(tk)
      if candidate_set and idx < max_seq_length - 1: # still not empty:
        random_candi_lst.append([idx, candidate_set])
    # random pick one to replace
    token_res = self(text,
                  add_special_tokens=True,
                  max_length=max_seq_length,
                  padding="max_length",
                  truncation=True,)
    if random_candi_lst:
      # random pick one
      replace_token = random.choice(random_candi_lst)
      split_tokens[replace_token[0]] = random.choice(list(replace_token[1]))
      token_res['input_ids'][replace_token[0] + 1] = self.vocab[split_tokens[replace_token[0]]]
      return split_tokens, token_res
    else:
      return None, token_res

  def _convert_token_to_id(self, token: str):
    """Converts a token (str/unicode) in an id using the vocab.

    Parameters
    ----------
    token: str
      String token from a larger sequence to be converted to a numerical id.
    """

    return self.vocab.get(token, self.vocab.get(self.unk_token))

  def _convert_id_to_token(self, index: int):
    """Converts an index (integer) in a token (string/unicode) using the vocab.

    Parameters
    ----------
    index: int
      Integer index to be converted back to a string-based token as part of a larger sequence.
    """

    return self.ids_to_tokens.get(index, self.unk_token)

  def convert_tokens_to_string(self, tokens: List[str]):
    """Converts a sequence of tokens (string) in a single string.

    Parameters
    ----------
    tokens: List[str]
      List of tokens for a given string sequence.

    Returns
    -------
    out_string: str
      Single string from combined tokens.
    """

    out_string: str = " ".join(tokens).replace(" ##", "").strip()
    return out_string

  def add_special_tokens_ids_single_sequence(self, token_ids: List[int]):
    """Adds special tokens to the a sequence for sequence classification tasks.

    A BERT sequence has the following format: [CLS] X [SEP]

    Parameters
    ----------
    token_ids: list[int]
      list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
    """

    return [self.cls_token_id] + token_ids + [self.sep_token_id]

  def add_special_tokens_single_sequence(self, tokens: List[str]):
    """Adds special tokens to the a sequence for sequence classification tasks.
    A BERT sequence has the following format: [CLS] X [SEP]

    Parameters
    ----------
    tokens: List[str]
      List of tokens for a given string sequence.
    """
    return [self.cls_token] + tokens + [self.sep_token]

  def add_special_tokens_ids_sequence_pair(self, token_ids_0: List[int],
                                           token_ids_1: List[int]) -> List[int]:
    """Adds special tokens to a sequence pair for sequence classification tasks.
    A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]

    Parameters
    ----------
    token_ids_0: List[int]
      List of ids for the first string sequence in the sequence pair (A).
    token_ids_1: List[int]
      List of tokens for the second string sequence in the sequence pair (B).
    """

    sep = [self.sep_token_id]
    cls = [self.cls_token_id]

    return cls + token_ids_0 + sep + token_ids_1 + sep

  def add_padding_tokens(self,
                         token_ids: List[int],
                         length: int,
                         right: bool = True) -> List[int]:
    """Adds padding tokens to return a sequence of length max_length.
    By default padding tokens are added to the right of the sequence.

    Parameters
    ----------
    token_ids: list[int]
      list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
    length: int
      TODO
    right: bool, default True
      TODO

    Returns
    -------
    List[int]
      TODO
    """
    padding = [self.pad_token_id] * (length - len(token_ids))

    if right:
      return token_ids + padding
    else:
      return padding + token_ids

  def save_vocabulary(
      self, vocab_path: str
  ):  # -> tuple[str]: doctest issue raised with this return type annotation
    """Save the tokenizer vocabulary to a file.

    Parameters
    ----------
    vocab_path: obj: str
      The directory in which to save the SMILES character per line vocabulary file.
      Default vocab file is found in deepchem/feat/tests/data/vocab.txt

    Returns
    -------
    vocab_file: Tuple
      Paths to the files saved.
      typle with string to a SMILES character per line vocabulary file.
      Default vocab file is found in deepchem/feat/tests/data/vocab.txt
    """
    index = 0
    if os.path.isdir(vocab_path):
      vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
    else:
      vocab_file = vocab_path
    with open(vocab_file, "w", encoding="utf-8") as writer:
      for token, token_index in sorted(
          self.vocab.items(), key=lambda kv: kv[1]):
        if index != token_index:
          logger.warning(
              "Saving vocabulary to {}: vocabulary indices are not consecutive."
              " Please check that the vocabulary is not corrupted!".format(
                  vocab_file))
          index = token_index
        writer.write(token + "\n")
        index += 1
    return (vocab_file,)
    
#VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}
class SmilesTokenizer(PreTrainedTokenizer):
  """
  Tokenizer in RobertaTokenizer style.
  Creates the SmilesTokenizer class. The tokenizer heavily inherits from the BertTokenizer
  implementation found in Huggingface's transformers library. It runs a WordPiece tokenization
  algorithm over SMILES strings using the tokenisation SMILES regex developed by Schwaller et. al.

  Please see https://github.com/huggingface/transformers
  and https://github.com/rxn4chemistry/rxnfp for more details.

  Examples
  --------
  >>> from deepchem.feat.smiles_tokenizer import SmilesTokenizer
  >>> current_dir = os.path.dirname(os.path.realpath(__file__))
  >>> vocab_path = os.path.join(current_dir, 'tests/data', 'vocab.txt')
  >>> tokenizer = SmilesTokenizer(vocab_path)
  >>> print(tokenizer.encode("CC(=O)OC1=CC=CC=C1C(=O)O"))
  [12, 16, 16, 17, 22, 19, 18, 19, 16, 20, 22, 16, 16, 22, 16, 16, 22, 16, 20, 16, 17, 22, 19, 18, 19, 13]


  References
  ----------
  .. [1] Schwaller, Philippe; Probst, Daniel; Vaucher, Alain C.; Nair, Vishnu H; Kreutter, David;
     Laino, Teodoro; et al. (2019): Mapping the Space of Chemical Reactions using Attention-Based Neural
     Networks. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.9897365.v3

  Note
  ----
  This class requires huggingface's transformers and tokenizers libraries to be installed.
  """
  vocab_files_names = VOCAB_FILES_NAMES

  def __init__(
      self,
      vocab_file: str = '',
      bos_token="<s>",
      eos_token="</s>",
      sep_token="</s>",
      cls_token="<s>",
      unk_token="<unk>",
      pad_token="<pad>",
      mask_token="<mask>",
      add_prefix_space=False,
      **kwargs):
    """Constructs a SmilesTokenizer.

    Parameters
    ----------
    vocab_file: str
      Path to a SMILES character per line vocabulary file.
      Default vocab file is found in deepchem/feat/tests/data/vocab.txt
    """

    bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
    eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
    sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
    cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
    unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
    pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

    # Mask token behave like a normal word, i.e. include the space before it
    mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

    super().__init__(
        vocab_file=vocab_file,
        bos_token=bos_token,
        eos_token=eos_token,
        unk_token=unk_token,
        sep_token=sep_token,
        cls_token=cls_token,
        pad_token=pad_token,
        mask_token=mask_token,
        add_prefix_space=add_prefix_space,
        **kwargs,
    )


    #super().__init__(vocab_file, **kwargs) #merges_file
    # take into account special tokens in max length
    # self.max_len_single_sentence = self.model_max_length - 2
    # self.max_len_sentences_pair = self.model_max_length - 3

    if not os.path.isfile(vocab_file):
      raise ValueError(
          "Can't find a vocab file at path '{}'.".format(vocab_file))
    with open(vocab_file, 'r') as vr:
        self.vocab = json.load(vr)
        # self.vocab = load_vocab(vocab_file)
    # self.highest_unused_index = max(
    #     [i for i, v in enumerate(self.vocab.keys()) if v.startswith("[unused")])
    self.ids_to_tokens = collections.OrderedDict(
        [(ids, tok) for tok, ids in self.vocab.items()])
    
    self.basic_tokenizer = smiles_tokenizer
    
    self.init_kwargs["model_max_length"] = self.model_max_length

  @property
  def vocab_size(self):
    return len(self.vocab)

  @property
  def vocab_list(self):
    return list(self.vocab.keys())

  def _tokenize(self, text: str):
    """Tokenize a string into a list of tokens.

    Parameters
    ----------
    text: str
      Input string sequence to be tokenized.
    """

    split_tokens = [token for token in self.basic_tokenizer(text)]
    return split_tokens

  def build_inputs_with_special_tokens(
      self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
      """
      Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
      adding special tokens. A RoBERTa sequence has the following format:

      - single sequence: ``<s> X </s>``
      - pair of sequences: ``<s> A </s></s> B </s>``

      Args:
          token_ids_0 (:obj:`List[int]`):
              List of IDs to which the special tokens will be added.
          token_ids_1 (:obj:`List[int]`, `optional`):
              Optional second list of IDs for sequence pairs.

      Returns:
          :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
      """
      if token_ids_1 is None:
          return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
      cls = [self.cls_token_id]
      sep = [self.sep_token_id]
      return cls + token_ids_0 + sep + sep + token_ids_1 + sep

  def _convert_token_to_id(self, token: str):
    """Converts a token (str/unicode) in an id using the vocab.

    Parameters
    ----------
    token: str
      String token from a larger sequence to be converted to a numerical id.
    """

    return self.vocab.get(token, self.vocab.get(self.unk_token))

  def _convert_id_to_token(self, index: int):
    """Converts an index (integer) in a token (string/unicode) using the vocab.

    Parameters
    ----------
    index: int
      Integer index to be converted back to a string-based token as part of a larger sequence.
    """

    return self.ids_to_tokens.get(index, self.unk_token)

  def convert_tokens_to_string(self, tokens: List[str]):
    """Converts a sequence of tokens (string) in a single string.

    Parameters
    ----------
    tokens: List[str]
      List of tokens for a given string sequence.

    Returns
    -------
    out_string: str
      Single string from combined tokens.
    """

    out_string: str = " ".join(tokens).replace(" ##", "").strip()
    return out_string

  def add_special_tokens_ids_single_sequence(self, token_ids: List[int]):
    """Adds special tokens to the a sequence for sequence classification tasks.

    A BERT sequence has the following format: [CLS] X [SEP]

    Parameters
    ----------
    token_ids: list[int]
      list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
    """

    return [self.cls_token_id] + token_ids + [self.sep_token_id]

  def add_special_tokens_single_sequence(self, tokens: List[str]):
    """Adds special tokens to the a sequence for sequence classification tasks.
    A BERT sequence has the following format: [CLS] X [SEP]

    Parameters
    ----------
    tokens: List[str]
      List of tokens for a given string sequence.
    """
    return [self.cls_token] + tokens + [self.sep_token]

  def add_special_tokens_ids_sequence_pair(self, token_ids_0: List[int],
                                           token_ids_1: List[int]) -> List[int]:
    """Adds special tokens to a sequence pair for sequence classification tasks.
    A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]

    Parameters
    ----------
    token_ids_0: List[int]
      List of ids for the first string sequence in the sequence pair (A).
    token_ids_1: List[int]
      List of tokens for the second string sequence in the sequence pair (B).
    """

    sep = [self.sep_token_id]
    cls = [self.cls_token_id]

    return cls + token_ids_0 + sep + token_ids_1 + sep

  def add_padding_tokens(self,
                         token_ids: List[int],
                         length: int,
                         right: bool = True) -> List[int]:
    """Adds padding tokens to return a sequence of length max_length.
    By default padding tokens are added to the right of the sequence.

    Parameters
    ----------
    token_ids: list[int]
      list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
    length: int
      TODO
    right: bool, default True
      TODO

    Returns
    -------
    List[int]
      TODO
    """
    padding = [self.pad_token_id] * (length - len(token_ids))

    if right:
      return token_ids + padding
    else:
      return padding + token_ids

  def save_vocabulary(
      self, vocab_path: str
  ):  # -> tuple[str]: doctest issue raised with this return type annotation
    """Save the tokenizer vocabulary to a file.

    Parameters
    ----------
    vocab_path: obj: str
      The directory in which to save the SMILES character per line vocabulary file.
      Default vocab file is found in deepchem/feat/tests/data/vocab.txt

    Returns
    -------
    vocab_file: Tuple
      Paths to the files saved.
      typle with string to a SMILES character per line vocabulary file.
      Default vocab file is found in deepchem/feat/tests/data/vocab.txt
    """
    index = 0
    if os.path.isdir(vocab_path):
      vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
    else:
      vocab_file = vocab_path
    with open(vocab_file, "w", encoding="utf-8") as writer:
      for token, token_index in sorted(
          self.vocab.items(), key=lambda kv: kv[1]):
        if index != token_index:
          logger.warning(
              "Saving vocabulary to {}: vocabulary indices are not consecutive."
              " Please check that the vocabulary is not corrupted!".format(
                  vocab_file))
          index = token_index
        writer.write(token + "\n")
        index += 1
    return (vocab_file,)



class SmilesIUPACTokenizer(PreTrainedTokenizer):
  """
  Tokenizer in RobertaTokenizer style.
  Creates the SmilesTokenizer class. The tokenizer heavily inherits from the BertTokenizer
  implementation found in Huggingface's transformers library. It runs a WordPiece tokenization
  algorithm over SMILES strings using the tokenisation SMILES regex developed by Schwaller et. al.

  Please see https://github.com/huggingface/transformers
  and https://github.com/rxn4chemistry/rxnfp for more details.

  Examples
  --------
  >>> from deepchem.feat.smiles_tokenizer import SmilesTokenizer
  >>> current_dir = os.path.dirname(os.path.realpath(__file__))
  >>> vocab_path = os.path.join(current_dir, 'tests/data', 'vocab.txt')
  >>> tokenizer = SmilesTokenizer(vocab_path)
  >>> print(tokenizer.encode("CC(=O)OC1=CC=CC=C1C(=O)O"))
  [12, 16, 16, 17, 22, 19, 18, 19, 16, 20, 22, 16, 16, 22, 16, 16, 22, 16, 20, 16, 17, 22, 19, 18, 19, 13]


  References
  ----------
  .. [1] Schwaller, Philippe; Probst, Daniel; Vaucher, Alain C.; Nair, Vishnu H; Kreutter, David;
     Laino, Teodoro; et al. (2019): Mapping the Space of Chemical Reactions using Attention-Based Neural
     Networks. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.9897365.v3

  Note
  ----
  This class requires huggingface's transformers and tokenizers libraries to be installed.
  """
  vocab_files_names = VOCAB_FILES_NAMES

  def __init__(
      self,
      vocab_file: str = '',
      bos_token="<s>",
      eos_token="</s>",
      sep_token="</s>",
      cls_token="<s>",
      unk_token="<unk>",
      pad_token="<pad>",
      mask_token="<mask>",
      add_prefix_space=False,
      **kwargs):
    """Constructs a SmilesTokenizer.

    Parameters
    ----------
    vocab_file: str
      Path to a SMILES character per line vocabulary file.
      Default vocab file is found in deepchem/feat/tests/data/vocab.txt
    """

    bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
    eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
    sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
    cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
    unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
    pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

    # Mask token behave like a normal word, i.e. include the space before it
    mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

    super().__init__(
        vocab_file=vocab_file,
        bos_token=bos_token,
        eos_token=eos_token,
        unk_token=unk_token,
        sep_token=sep_token,
        cls_token=cls_token,
        pad_token=pad_token,
        mask_token=mask_token,
        add_prefix_space=add_prefix_space,
        **kwargs,
    )


    #super().__init__(vocab_file, **kwargs) #merges_file
    # take into account special tokens in max length
    # self.max_len_single_sentence = self.model_max_length - 2
    # self.max_len_sentences_pair = self.model_max_length - 3

    if not os.path.isfile(vocab_file):
      raise ValueError(
          "Can't find a vocab file at path '{}'.".format(vocab_file))
    with open(vocab_file, 'r') as vr:
        self.vocab = json.load(vr)
        # self.vocab = load_vocab(vocab_file)
    # self.highest_unused_index = max(
    #     [i for i, v in enumerate(self.vocab.keys()) if v.startswith("[unused")])
    self.ids_to_tokens = collections.OrderedDict(
        [(ids, tok) for tok, ids in self.vocab.items()])
    
    self.smiles_basic_tokenizer = smiles_tokenizer
    self.iupac_basic_tokenizer = iupac_tokenizer
    self.input_iupac = False
    
    self.init_kwargs["model_max_length"] = self.model_max_length

  def set_inpuac_input(self, is_iupac):
    self.input_iupac = is_iupac

  @property
  def vocab_size(self):
    return len(self.vocab)

  @property
  def vocab_list(self):
    return list(self.vocab.keys())

  def _tokenize(self, text: str):
    """Tokenize a string into a list of tokens.

    Parameters
    ----------
    text: str
      Input string sequence to be tokenized.
    """
    if self.input_iupac:
      split_tokens = [token for token in self.iupac_basic_tokenizer(text, is_check=False)]
    else:
      split_tokens = [token for token in self.smiles_basic_tokenizer(text)]
    return split_tokens

  def build_inputs_with_special_tokens(
      self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
      """
      Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
      adding special tokens. A RoBERTa sequence has the following format:

      - single sequence: ``<s> X </s>``
      - pair of sequences: ``<s> A </s></s> B </s>``

      Args:
          token_ids_0 (:obj:`List[int]`):
              List of IDs to which the special tokens will be added.
          token_ids_1 (:obj:`List[int]`, `optional`):
              Optional second list of IDs for sequence pairs.

      Returns:
          :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
      """
      if token_ids_1 is None:
          return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
      cls = [self.cls_token_id]
      sep = [self.sep_token_id]
      return cls + token_ids_0 + sep + sep + token_ids_1 + sep

  def _convert_token_to_id(self, token: str):
    """Converts a token (str/unicode) in an id using the vocab.

    Parameters
    ----------
    token: str
      String token from a larger sequence to be converted to a numerical id.
    """

    return self.vocab.get(token, self.vocab.get(self.unk_token))

  def _convert_id_to_token(self, index: int):
    """Converts an index (integer) in a token (string/unicode) using the vocab.

    Parameters
    ----------
    index: int
      Integer index to be converted back to a string-based token as part of a larger sequence.
    """

    return self.ids_to_tokens.get(index, self.unk_token)

  def convert_tokens_to_string(self, tokens: List[str]):
    """Converts a sequence of tokens (string) in a single string.

    Parameters
    ----------
    tokens: List[str]
      List of tokens for a given string sequence.

    Returns
    -------
    out_string: str
      Single string from combined tokens.
    """

    out_string: str = " ".join(tokens).replace(" ##", "").strip()
    return out_string

  def add_special_tokens_ids_single_sequence(self, token_ids: List[int]):
    """Adds special tokens to the a sequence for sequence classification tasks.

    A BERT sequence has the following format: [CLS] X [SEP]

    Parameters
    ----------
    token_ids: list[int]
      list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
    """

    return [self.cls_token_id] + token_ids + [self.sep_token_id]

  def add_special_tokens_single_sequence(self, tokens: List[str]):
    """Adds special tokens to the a sequence for sequence classification tasks.
    A BERT sequence has the following format: [CLS] X [SEP]

    Parameters
    ----------
    tokens: List[str]
      List of tokens for a given string sequence.
    """
    return [self.cls_token] + tokens + [self.sep_token]

  def add_special_tokens_ids_sequence_pair(self, token_ids_0: List[int],
                                           token_ids_1: List[int]) -> List[int]:
    """Adds special tokens to a sequence pair for sequence classification tasks.
    A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]

    Parameters
    ----------
    token_ids_0: List[int]
      List of ids for the first string sequence in the sequence pair (A).
    token_ids_1: List[int]
      List of tokens for the second string sequence in the sequence pair (B).
    """

    sep = [self.sep_token_id]
    cls = [self.cls_token_id]

    return cls + token_ids_0 + sep + token_ids_1 + sep

  def add_padding_tokens(self,
                         token_ids: List[int],
                         length: int,
                         right: bool = True) -> List[int]:
    """Adds padding tokens to return a sequence of length max_length.
    By default padding tokens are added to the right of the sequence.

    Parameters
    ----------
    token_ids: list[int]
      list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
    length: int
      TODO
    right: bool, default True
      TODO

    Returns
    -------
    List[int]
      TODO
    """
    padding = [self.pad_token_id] * (length - len(token_ids))

    if right:
      return token_ids + padding
    else:
      return padding + token_ids

  def save_vocabulary(
      self, vocab_path: str
  ):  # -> tuple[str]: doctest issue raised with this return type annotation
    """Save the tokenizer vocabulary to a file.

    Parameters
    ----------
    vocab_path: obj: str
      The directory in which to save the SMILES character per line vocabulary file.
      Default vocab file is found in deepchem/feat/tests/data/vocab.txt

    Returns
    -------
    vocab_file: Tuple
      Paths to the files saved.
      typle with string to a SMILES character per line vocabulary file.
      Default vocab file is found in deepchem/feat/tests/data/vocab.txt
    """
    index = 0
    if os.path.isdir(vocab_path):
      vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
    else:
      vocab_file = vocab_path
    with open(vocab_file, "w", encoding="utf-8") as writer:
      for token, token_index in sorted(
          self.vocab.items(), key=lambda kv: kv[1]):
        if index != token_index:
          logger.warning(
              "Saving vocabulary to {}: vocabulary indices are not consecutive."
              " Please check that the vocabulary is not corrupted!".format(
                  vocab_file))
          index = token_index
        writer.write(token + "\n")
        index += 1
    return (vocab_file,)


#from deepchem.feat.smiles_tokenizer import SmilesTokenizer # not using it
from transformers import RobertaTokenizer

if __name__ == '__main__':
    # iupac_str = '3-acetyloxy-4-(trimethylazaniumyl)butanoate'
    # iupac_str = '2-hydroxy-6-oxonona-2,4-dienedioic acid'
    # res = iupac_tokenizer(iupac_str)
    # print(res)

    # ## getting smiles vocabulary
    # smiles_list = []
    # df_smiles = pd.read_csv('/sharefs/ylx/chem_data/pubchem/data_1m/smiles.csv')
    # smiles_list = df_smiles['Canonical'].tolist()

    # print(len(smiles_list)) #896718

    # smiles_model = SmilesModel(smiles_list,nworkers=8)
    # print(len(smiles_model.vocab)) # 401
    # print(smiles_model.vocab)

    # ## getting smiles vocabulary
    # tok = RobertaTokenizer.from_pretrained('seyonec/SMILES_tokenized_PubChem_shard00_160k')
    # tok_vocab = list((tok.get_vocab()).keys())[16:-2]
    # print(len(tok_vocab)) # 
    # print(tok_vocab)
    # tok_vocab = set(tok_vocab)
    # special_tokens =  ["<s>","<pad>","</s>","<unk>","<mask>", ';', '.', '>']
    # nspec = len(special_tokens)

    # vocab = {}
    # for i,spec in enumerate(special_tokens):
    #     vocab[spec] = i
    # vocab.update(dict(zip(sorted(tok_vocab),
    #                       range(nspec, nspec+len(tok_vocab)))))
    # rev_vocab = reverse_vocab(vocab)
    # vocsize = len(vocab)

    # with open('/sharefs/ylx/chem_data/results/tokenizer_dir/smiles_regex/vocab.json','w') as w:
    #     json.dump(vocab,w)


    ### getting iupac vocabulary
    # iupac_list = []
    # df_iupac = pd.read_csv('/sharefs/ylx/chem_data/pubchem/data_1m/iupacs.csv')
    # iupac_list = df_iupac['Preferred'].tolist()

    # print(len(iupac_list)) #896718

    # iupac_model = IupacModel(iupac_list,nworkers=8)
    # print(len(iupac_model.vocab)) # 671
    # print(iupac_model.vocab)

    ### tokenize smiles 
    input_smiles = "C1=CC(=C(C=C1[N+](=O)[O-])[N+](=O)[O-])Cl"
    #vocab_path = "Datas/tokenizer_dir/smiles_reg/smiles_vob.txt"
    vocab_path = "./Datas/tokenizer_dir/smiles_regex/"
    tokenizer = SmilesTokenizer.from_pretrained(vocab_path)
    output = tokenizer.tokenize(input_smiles)
    print(output) # ['C', '1', '=', 'C', 'C', '(', '=', 'C', '(', 'C', '=', 'C', '1', '[N+]', '(', '=', 'O', ')', '[O-]', ')', '[N+]', '(', '=', 'O', ')', '[O-]', ')', 'Cl']
    output = tokenizer(input_smiles,
                add_special_tokens=True,
                max_length=32,
                padding="max_length",
                truncation=True,)
    print(output)
    '''
    {'input_ids': [0, 43, 30, 39, 43, 43, 24, 39, 43, 24, 43, 39, 43, 30, 327, 24, 39, 48, 25, 368, 25, 327, 24, 39, 48, 25, 368, 25, 44, 2, 1, 1], 
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]}
    '''
    
    ### tokenize iupac 
    iupac_str = '3-acetyloxy-4-(trimethylazaniumyl)butanoate'
    vocab_path = "./Datas/tokenizer_dir/iupac_regex/"#vocab.json" 
    #assert os.path.isdir(vocab_path)
    tokenizer = IUPACTokenizer.from_pretrained(vocab_path)
    output = tokenizer.tokenize(iupac_str)
    print(output) # ['3', '-', 'acetyl', 'oxy', '-', '4', '-', '(', 'tri', 'methyl', 'azanium', 'yl', ')', 'but', 'an', 'oate']
    output = tokenizer(iupac_str,
                add_special_tokens=True,
                max_length=32,
                padding="max_length",
                truncation=True,)
    print(output)
    '''
    {'input_ids': [0, 48, 24, 140, 475, 24, 59, 24, 10, 652, 425, 177, 664, 22, 217, 155, 453, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

    '''

    

    
    
    