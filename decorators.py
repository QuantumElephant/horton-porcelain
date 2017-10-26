from functools import wraps


#
# Decorators
#

def onetime(varname: str):
    """
    Prevents basic Python types from being changed after setting.
    Numpy arrays also need the read-only flag set on them.
    For use with Python setters, must be decorated before (closer to the function) @setter is
    called.

    Parameters
    ----------
    varname
        Name of the variable being tested. It does not need to be defined at interpretation
        time.

    Returns
    -------

    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, varname, None) is not None:
                print("Trying to set a one-time attribute {varname}. Ignored")
                return
            func(self, *args, **kwargs)

        return wrapper

    return decorator


def cache(varname: str):
    """
    Returns a variable if it is already defined. Otherwise, it calls the code in the property.
    Must be decorated before (closer to the function) @property.

    Parameters
    ----------
    varname
        Name of variable to cache. It does not need to be defined at interpretation.

    Returns
    -------

    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            val = getattr(self, varname, None)
            if val is None:
                val = func(self, *args, **kwargs)
                setattr(self, varname, val)
            return val

        return wrapper

    return decorator


def delayed(func):
    """
    Safety check to make sure the instance has the second stage of instantiation.
    The function decorated will return AttributeError if the function with @finalize has not
    been called yet.

    Parameters
    ----------
    func

    Returns
    -------

    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "init_finished", False):
            print("Instance must finalize instantiation before calling compute functions.")
            raise AttributeError
        return func(self, *args, **kwargs)

    return wrapper


def finalize(func):
    """
    When the function decorated completes, the instance will be marked as fully instantiated.

    Parameters
    ----------
    func

    Returns
    -------

    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self.init_finished = True

    return wrapper
