#!/usr/bin/env python3
"""Inspect CUAD dataset item structure and print examples to stdout."""
from datasets import load_dataset
import json

def main():
    ds = load_dataset('Nadav-Timor/CUAD', split='train')
    print('dataset_len=', len(ds))
    for i in range(3):
        item = ds[i]
        print('\n=== ITEM', i, '===')
        print('keys:', list(item.keys()))
        print('types:', {k: type(v).__name__ for k,v in item.items()})
        # pretty print a few fields
        for k in ('contract_text','context','qas','question','answers','id'):
            if k in item:
                print(f'---- {k} ----')
                try:
                    print(json.dumps(item[k], ensure_ascii=False, indent=2)[:1000])
                except Exception:
                    print(repr(item[k])[:1000])

if __name__ == '__main__':
    main()





