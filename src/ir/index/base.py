import collections
from typing import Any, Protocol

SearchResults = collections.namedtuple("SearchResults", ["ids", "scores"])


class Index(Protocol):
    """Abstract vector store protocol."""
    def __init__(self, **kwargs):
        pass

    def load_dstore(self, file, **kwargs):
        pass

    def load_vstore(self, file, **kwargs):
        pass

    def search(self, q_embs: Any, k: int, **kwargs) -> SearchResults:
        pass
