
from hparams import symbols

print("MY SYMBOLS",len(symbols))

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def text_to_sequence(text):

    text_enc=[_symbol_to_id[s] for s in text if s in _symbol_to_id]
    add_eos_to_text=True
    prepend_space_to_text=True

    if prepend_space_to_text:
        text_enc.insert(0, _symbol_to_id[' '])

    if add_eos_to_text:
        text_enc.append(_symbol_to_id['E'])
    return text_enc