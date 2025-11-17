from typing import Tuple
import opt_einsum as oe  # type: ignore[import-untyped]
from opt_einsum.parser import _einsum_symbols_base # type: ignore[import-untyped]
# from .mpsqsc.mpstate import MPState
# import torch



class GetSymbolFn:

    def __init__(self):
        self.symbol_fn = oe.get_symbol
        self._symbol_count = 0

    def __call__(self) -> str:
        symbol = self.symbol_fn(self._symbol_count)
        self._symbol_count += 1
        return symbol
    
    def __getstate__(self) -> dict:
        return {"symbol_fn": self.symbol_fn, "_symbol_count": self._symbol_count}
    
    @property
    def symbol_count(self) -> int:
        return self._symbol_count
    
    @staticmethod
    def get_symbol(i: int) -> str:
        return oe.get_symbol(i)

    # get the index of the symbol
    @staticmethod
    def inv_get_symbol(s: str) -> int:
        """Inverse of get_symbol(i). Returns the integer i such that get_symbol(i) == s."""
        if len(s) != 1:
            raise ValueError("Symbol must be a single character")

        # Case 1: ASCII base symbols (i < 52)
        if s in _einsum_symbols_base:
            return _einsum_symbols_base.index(s)

        code = ord(s)

        # Case 3: i >= 55296 → code = i + 2048
        # That means code >= 55296 + 2048 = 57344
        if code >= 57344:
            i = code - 2048
            if i >= 55296:
                return i

        # Case 2: 52 ≤ i < 55296 → code = i + 140
        i = code - 140
        if 52 <= i < 55296:
            return i

        raise ValueError(f"Symbol {s!r} is not a valid output of get_symbol.")
    
    def reset(self):
        self._symbol_count = 0
    
    def shift_chars(self, shift: int, *args):
        """
        1. Shift `symbol_count` by `shift`.
        2. For each arg (a container of containers of ... of str), shift each
        symbol s to get_symbol(inv_get_symbol(s) + shift), preserving structure.

        `s` can be:
            - a single-character symbol (e.g. 'a', 'β'), or
            - an einsum-style equation string (e.g. 'af,fbg,gch,hdi,ie->abcde').

        Returns the shifted version(s) of the provided args. Arguments
        themselves are not modified in-place.
        """

        # helper to shift a single symbol or an einsum equation
        def _shift_symbol(sym: str) -> str:
            # Case 1: einsum-style equation string – shift each index char
            # Example: "af,fbg,gch,hdi,ie->abcde"
            if "->" in sym or "," in sym:
                def _shift_char(c: str) -> str:
                    # punctuation separators: leave unchanged
                    if c in ",-> ":
                        return c
                    # everything else is treated as an index symbol
                    i = self.inv_get_symbol(c)
                    j = i + shift
                    if j < 0:
                        raise ValueError(
                            f"Shift would make symbol index negative: {i} + {shift}"
                        )
                    return self.get_symbol(j)

                return "".join(_shift_char(c) for c in sym)

            # Case 2: single-character symbol (original behavior)
            if len(sym) != 1:
                raise ValueError(f"Expected single-character symbol or einsum eq, got {sym!r}")
            i = self.inv_get_symbol(sym)
            j = i + shift
            if j < 0:
                raise ValueError(
                    f"Shift would make symbol index negative: {i} + {shift}"
                )
            return self.get_symbol(j)

        # recursive helper to walk through nested containers
        def _shift_container(obj):
            # leaf: string (either a single symbol or an equation)
            if isinstance(obj, str):
                return _shift_symbol(obj)

            # list, tuple, set, dict – recurse and rebuild same type
            if isinstance(obj, list):
                return [_shift_container(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_shift_container(x) for x in obj)
            if isinstance(obj, set):
                return {_shift_container(x) for x in obj}
            if isinstance(obj, dict):
                return {k: _shift_container(v) for k, v in obj.items()}

            # anything else is returned unchanged (e.g. ints, None, etc.)
            return obj

        # apply to all args and return
        if len(args) == 0:
            return None
        if len(args) == 1:
            return _shift_container(args[0])
        return tuple(_shift_container(a) for a in args)

    


        # build the state equation.

    
class _EqAndPathCache:
    """
    Small helper to cache equation string + opt_einsum path
    keyed by the shapes of the current tensors.
    """
    def __init__(self):
        self.eq: str | None = None
        self.path: oe.Path | None = None
        self.shape_sig: Tuple[Tuple[int, ...], ...] | None = None

    def invalidate(self):
        self.eq, self.path, self.shape_sig = None, None, None
