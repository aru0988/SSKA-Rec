from .vocabulary import AliCCP_Vocabulary_Size

DATA_FILES = {
    "train": "AliCCP/ctr_cvr.train",
    "val": "AliCCP/ctr_cvr.dev",
    "test": "AliCCP/ctr_cvr.test",
}


def build_vocab() -> dict:
    """Return feature cardinalities with user ID and the target field excluded."""
    vocab = AliCCP_Vocabulary_Size.copy()
    for key in ("101", "301"):
        vocab.pop(key)
    return vocab
