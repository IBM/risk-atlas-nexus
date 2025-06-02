from random import randint


def type_check(log_code, *types, allow_none=False, **variables):
    """Check for acceptable types for a given object.

    Args:
        log_code:  str
            A log code in str format.
        *types:  type or None
           The acceptable types for `variables`.
        allow_none:  bool
            If `True` then the values of `variables` are allowed to take on the value of `None` without causing the
            type check to fail.
        **variables:  object
            The keyword arguments to be examined for acceptable type.

    """

    if not types:
        raise RuntimeError("Invalid type check: no types specified")

    if not variables:
        raise RuntimeError("Invalid type check: no variables specified")

    for name, variable in variables.items():
        if allow_none and variable is None:
            continue

        # check if variable is an instance of one of `types`
        if not isinstance(variable, types):
            type_name = type(variable).__name__
            valid_type_names = tuple(typ.__name__ for typ in types)
            if allow_none:
                valid_type_names += (type(None).__name__,)

            # create, log and raise an appropriate exception
            raise TypeError(
                "{} type check failed: variable `{}` has type `{}` not in `{}`".format(
                    log_code, name, type_name, valid_type_names
                )
            )


def value_check(log_code, condition, msg="", *args):
    """Check for acceptable values for a given object.

    Args:
        log_code:  str
            A log code in str format
        condition:  bool
            A boolean value that should describe if this check passes `True` or fails `False`.
            Upon calling this function, this is typically provided as an expression, e.g.,
            `0 < variable < 1`.
        msg:  str
            A string message describing the value check that failed.  If the empty string is
            provided (default) then no additional information will be provided.
        *args:
            Additional `{}`-style string formatting arguments.
    """

    if not condition:
        interpolated_msg = msg.format(*args)
        raise ValueError("value check failed:{} {}".format(log_code, interpolated_msg))


def _gen_new_error_code(prefix="RAN", level="error"):
    """Internal helper to make a new error code.

    Args:
        prefix:  str
            A error code prefix in str format
        level:  str

    """
    level_letter = "E"
    if level == "error":
        level_letter = "E"
    if level == "warning":
        level_letter = "W"

    return prefix + "{:08d}".format(randint(0, 10**8)) + level_letter
