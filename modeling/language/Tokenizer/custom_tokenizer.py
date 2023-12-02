import torch
import torch.nn as nn


def split_by_ordered_substrings(sentence, substrings):
    results = []
    substring_indices = []

    start_index = 0
    for i, substring in enumerate(substrings):
        # Find the start of the substring in the remaining part of the sentence
        index = sentence[start_index:].find(substring)

        if index == -1:
            continue

        # Append any text before the substring to the results, including spaces
        if index > 0:
            results.append(sentence[start_index:start_index+index])
            substring_indices.append(None)  # No match in the `substrings` list for this segment
        
        # Append the substring to the results
        results.append(substring)
        substring_indices.append(i)  # Append the index from the `substrings` list
        start_index += index + len(substring)

    # If there's any remaining part of the sentence after all substrings, append it to the results
    if start_index < len(sentence):
        results.append(sentence[start_index:])
        substring_indices.append(None)  # No match in the `substrings` list for this segment

    return results, substring_indices

class CustomCLIPTokenizer(nn.Module):
    def __init__(self, tokenizer, max_token_num):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_token_num = max_token_num
    
    def forward(self, sentence, entity_list):
        substrings, index = split_by_ordered_substrings(sentence, entity_list)
        tokens = self.tokenizer(
            substrings, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
        )

        _input_ids, _attention_mask = [], []
        start_end_condition = torch.zeros(len(entity_list), len(tokens['input_ids'][0])).bool()

        start_idx = 0
        end_idx = 0
        for j, (input_id, attn_mask, idx) in enumerate(zip(tokens['input_ids'], tokens['attention_mask'], index)):
            if j == 0:
                _input_ids += [input_id[attn_mask.bool()][:-1]]
                _attention_mask += [attn_mask[attn_mask!=0][:-1]]
            elif j == len(tokens['input_ids']) - 1:
                _input_ids += [input_id[attn_mask.bool()][1:]]
                _attention_mask += [attn_mask[attn_mask!=0][1:]]
            else:
                _input_ids += [input_id[attn_mask.bool()][1:-1]]
                _attention_mask += [attn_mask[attn_mask!=0][1:-1]]

            end_idx += len(_input_ids[-1])

            if idx is not None:
                start_end_condition[idx,start_idx:end_idx] = True
            start_idx = end_idx

        _input_ids_all = torch.ones(self.max_token_num, dtype=torch.long) * self.tokenizer.pad_token_id
        _attention_mask_all = torch.zeros(self.max_token_num, dtype=torch.long)

        _input_ids = torch.cat(_input_ids)[:self.max_token_num]
        _attention_mask = torch.cat(_attention_mask)[:self.max_token_num]

        _input_ids_all[:len(_input_ids)] = _input_ids
        _attention_mask_all[:len(_attention_mask)] = _attention_mask
        if len(_attention_mask) < self.max_token_num:
            _attention_mask_all[len(_attention_mask)] = 1

        tokens = {"input_ids": _input_ids_all[None,], "attention_mask": _attention_mask_all[None,]}
        return tokens, start_end_condition

