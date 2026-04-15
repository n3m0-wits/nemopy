"""Operator override logic for the vec library.

Element-wise operators block broadcasting; ``@`` follows NumPy rules.
"""


def _check_broadcast(a, b):
    """Raise :class:`ShapeError` if *a* and *b* would broadcast unsafely.

    Parameters
    ----------
    a, b : _VecBase
        Operands to validate.
    """
    raise NotImplementedError
