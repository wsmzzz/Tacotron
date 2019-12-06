import re
import  numpy as np
from . import cleaners
from .symbols import symbols
from .Mandarin_phone import phone_dict,rhythm_dict

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')
pattern=re.compile('[#*^&]')


def text_to_sequence(text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''

  rhythm_embedding=[]
  text=text.strip().split(' ')

  for index,i in enumerate(text):
    if pattern.findall(i):
      rhythm_embedding.append(rhythm_dict[text[index][-1]])
      text[index]=text[index][:-1]
    else:
      rhythm_embedding.append(0)

  # Check for curly braces and treat their contents as ARPAbet:
  phone_embeding=np.reshape(np.asarray([phone_dict[i] for i in text]),[-1,1])
  rhythm_embedding=np.reshape(np.asarray(rhythm_embedding),[-1,1])
  _=np.concatenate([phone_embeding,rhythm_embedding],axis=1)           #[T_x,2]

  return _


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'
