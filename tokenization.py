
import collections
import copy

from torch import Tensor
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    type = set()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
        if ':' not in token:
            type.add('special_token')
        else:
            type.add(token.split(':')[0])
    type = sorted(list(type))
    type = {t:index for index, t in enumerate(list(type))}
    return vocab, type


class MyTokenizer():
    def __init__(
            self,
            vocab_file,
            sep_token="[SEP]",
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            treatment_list = None,
            baseline_window = 90,
            fix_window_length=30,
    ):

        self.sep_token = sep_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        self.vocab, self.type = load_vocab(vocab_file)
        self.id2token = {index:token for token, index in self.vocab.items()}
        self.id2type = {index:type for type, index in self.type.items()}

        self.treatment_ids = None
        self.baseline_window = baseline_window
        self.fix_window_length = fix_window_length
        self.treatment_list = treatment_list

    def convert_token_ids_to_token_type_ids(self, token_ids):

        if isinstance(token_ids, int):
            token = self.id2token.get(token_ids)
            token_type_ids = self.type.get(token.split(':')[0]) if ':' in token else self.type.get('special_token')
        else:
            token_type_ids = []
            for token_id in token_ids:
                token = self.id2token.get(token_id)
                type_id = self.type.get(token.split(':')[0]) if ':' in token else self.type.get('special_token')
                token_type_ids.append(type_id)

        return token_type_ids

    def encode(self, data, max_length=None, padding=True, return_tensor=False):

        treatment_group = data['treatment_group']
        covariates = data['covariates']
        covariates_time = data['covariates_time']

        # [cls] treatment [sep] demo x1, x2, ...
        input_ids = [self.vocab.get(self.cls_token)]
        token_type_ids = [self.type.get('special_token')]
        treatment_list = self.treatment_list

        if treatment_list:
            # treatment and [sep]
            input_ids.append(self.vocab.get('medication:{}'.format(treatment_group)))
            token_type_ids.append(self.type.get('medication'))

            input_ids.append(self.vocab.get(self.sep_token))
            token_type_ids.append(self.type.get('special_token'))

        visit_time_ids = [0] * len(input_ids)
        physical_time_ids = [0] * len(input_ids)

        padding_idx = self.vocab.get(self.pad_token)

        n_visit = 0
        prev_visit = 0
        for i, covariate in enumerate(covariates):

            visit_time = covariates_time[i]

            if visit_time >= self.baseline_window :
                continue

            if covariate == 'medication:{}'.format(treatment_group):
                continue


            if visit_time != 0 and visit_time != prev_visit:
                n_visit += 1
                prev_visit = visit_time

            visit_time_ids.append(n_visit)
            physical_time_ids.append(visit_time//self.fix_window_length)

            if len(covariate.split(':')) > 2:
                covariate_type = covariate.split(':')[0]
                covariate_value = ':'.join(covariate.split(':')[1:])
            else:
                covariate_type, covariate_value = covariate.split(':')

            if ':' in covariate_value:
                covariate_value = covariate_value.replace(':', '_')

            if covariate_type in ['msa', 'region', 'industry']:
                continue

            token = '{}:{}'.format(covariate_type, covariate_value)
            if token not in self.vocab:
                if '/' in token:
                    token = token.replace('/', '&&')
                if ' ' in token:
                    token = token.replace(' ', '_')



            input_id = self.vocab.get(token)
            assert input_id
            input_ids.append(input_id)
            token_type_ids.append(self.type.get(covariate_type))


        # truncate
        if max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            token_type_ids = token_type_ids[:max_length]
            visit_time_ids = visit_time_ids[:max_length]
            physical_time_ids = physical_time_ids[:max_length]

        attention_mask = [1] * len(input_ids)
        # padding
        if padding:
            attention_mask += [padding_idx] * (max_length - len(input_ids))
            input_ids += [padding_idx] * (max_length - len(input_ids))
            token_type_ids += [self.type.get('special_token')] * (max_length - len(token_type_ids))
            visit_time_ids += [visit_time_ids[-1]] * (max_length - len(visit_time_ids))
            physical_time_ids += [physical_time_ids[-1]] * (max_length - len(physical_time_ids))

        treatment_label = None
        input_ids_cf = None
        if treatment_list:
            treatment_label = treatment_list.index(treatment_group)
            treatment_id = self.vocab.get('medication:{}'.format(treatment_group))
            cf_treatment_id = self.vocab.get('medication:{}'.format(treatment_list[1-treatment_label]))
            input_ids_cf = [id if id != treatment_id else cf_treatment_id for id in input_ids]

        return {'input_ids': input_ids,
                'input_ids_cf': input_ids_cf,
                'token_type_ids': token_type_ids,
                'visit_time_ids': visit_time_ids,
                'physical_time_ids': physical_time_ids,
                'attention_mask': attention_mask,
                'treatment_group': treatment_group,
                'treatment_label': treatment_label}


    def decode(self, token_ids, with_special_tokens=True):
        if isinstance(token_ids,Tensor):
            token_ids = token_ids.numpy()
        result = []

        if with_special_tokens:
            for id in token_ids:
                result.append(self.id2token.get(id))
        else:
            for id in token_ids:
                if self.id2token.get(id) in [self.sep_token, self.pad_token, self.cls_token, self.mask_token]:
                    continue
                result.append(self.id2token.get(id))

        return result


    def get_treatment_ids(self) -> List[int]:
        if not self.treatment_ids:
            self.treatment_ids = [v for k,v in self.vocab.items() if 'treatment' in k]

        return self.treatment_ids

    @property
    def all_special_tokens(self) -> List[str]:
        """
        `List[str]`: All the special tokens (`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        Convert tokens of `tokenizers.AddedToken` type to string.
        """
        all_toks = [self.sep_token,self.unk_token,self.pad_token,self.cls_token,self.mask_token]
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token) for token in tokens]

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.
        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        assert already_has_special_tokens and token_ids_1 is None, (
            "You cannot use ``already_has_special_tokens=False`` with this tokenizer. "
            "Please use a slow (full python) tokenizer to activate this argument. "
            "Or set `return_special_tokens_mask=True` when calling the encoding method "
            "to get the special tokens mask in any tokenizer. "
        )

        all_special_ids = self.all_special_ids  # cache the property
        all_treatment_ids = self.get_treatment_ids()

        combined_ids = all_special_ids + all_treatment_ids

        special_tokens_mask = [1 if token in combined_ids else 0 for token in token_ids_0]

        return special_tokens_mask

    def __len__(self):
        return len(self.vocab)
