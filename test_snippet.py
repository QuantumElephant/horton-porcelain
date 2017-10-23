from functools import wraps


def cache(varname):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, varname) is None:
                val = func(self, *args, **kwargs)
                setattr(self, varname, val)
            return getattr(self, varname)
        return wrapper
    return decorator

# def onetime(varname):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(self, *args, **kwargs):
#             if getattr(self, varname) is not None:
#                 print("Trying to set a one-time attribute {varname}. Ignored")
#                 return
#             func(self, *args, **kwargs)
#         return wrapper
#     return decorator


def delayed(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.init_finished:
            print("Instance must finalize instantiation before calling compute functions.")
            raise AttributeError
        return func(self, *args, **kwargs)
    return wrapper


def finalize(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self.init_finished = True
    return wrapper


class A:
    init_finished = False
    _v = "hi"

    @finalize
    def final(self):
        pass

    @property
    @delayed
    @cache("_v")
    def v(self):
        print("eval")
        return self._v


i = A()
# print(i.v)
i.final()
print(i.v)

