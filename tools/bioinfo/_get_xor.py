from typing import Tuple

def get_xor(list_a: list, list_b: list) -> Tuple[list]:
    overlap = set(list_a) & set(list_b)
    return list(set(list_a) - overlap), list(set(list_b) - overlap)
