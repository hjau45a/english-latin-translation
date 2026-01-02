import transformers
from typing import Any

def translate_en_lat(text: str, tokenizer: Any, model: Any) -> str:
    """ Translate an input text.
    This method handles tokenization and the actual translation.

    Note, for the sake of simplicity, we use defaults that are not appropriate
    for long translation sequences - generation max_length is fixed at 256
    tokens for example.
    Also note, the translation is only English -> Latin.
    
    Params:
        text: The input text which is assumed to be english.
        tokenizer: The tokenizer.
        model: The model.

    Returns:
        A string being the Latin equivalent of the English sentence in text.
    """
    tokenizer.src_lang = "eng_Latn"

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to('cuda')

    forced_bos_token_id = tokenizer.convert_tokens_to_ids('lat_Latn')

    output_tokens = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=256
    )
    
    translation = tokenizer.batch_decode(
        output_tokens,
        skip_special_tokens=True
    )[0]

    return translation
