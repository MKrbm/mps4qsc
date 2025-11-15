
import opt_einsum as oe # type: ignore[import-untyped]
import random
from qmpsqsc.models.opt_einsum_utils import GetSymbolFn


def test_get_symbol():
    """
    Check that the symbols are generated correctly for GetSymbolFn.
    """
    sym = GetSymbolFn()
    assert sym() == "a"
    assert sym() == "b"
    assert sym() == "c"
    assert sym() == "d"
    assert sym() == "e"
    assert sym() == "f"
    assert sym() == "g"

def test_inv_get_symbol():
    """
    Check that the inverse of get_symbol is correct for GetSymbolFn.
    """
    sym = GetSymbolFn()

    for _ in range(100):
        i = random.randint(0, 10000)
        symbol = sym.get_symbol(i)
        assert sym.inv_get_symbol(symbol) == i

def test_shift_symbols():
    """
    Check that the symbols are shifted correctly for GetSymbolFn.
    """
    sym = GetSymbolFn()
    symbols = ["a", "b", "c", "d", "e", "f", "g"]
    shifted_symbols = sym.shift_chars(1, symbols)
    assert shifted_symbols == ["b", "c", "d", "e", "f", "g", "h"]

    random_integers = [random.randint(1, 10000) for _ in range(100)]
    syms = [sym.get_symbol(i) for i in random_integers]

    shift = 100
    shifted_symbols = sym.shift_chars(shift, syms)

    back_shifted_symbols = sym.shift_chars(-shift, shifted_symbols)

    assert back_shifted_symbols == syms

    


