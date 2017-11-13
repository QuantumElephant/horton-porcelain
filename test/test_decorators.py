from numpy.testing import assert_raises

from decorators import onetime, cache, delayed, finalize


def test_onetime():
    class A:
        _x = 5

        @property
        def x(self):
            return self._x

        @x.setter
        @onetime("_x")
        def x(self, y):
            self._x = y

    a = A()
    a.x = 10
    assert a.x == 5


def test_cache():
    class A:
        _x = 5

        @property
        @cache("_x")
        def x(self):
            return 10

    a = A()
    assert a.x == 5


def test_delayed_finalize():
    class A:
        @finalize
        def finish(self):
            pass

        @delayed
        def do(self):
            pass

    a = A()
    with assert_raises(AttributeError):
        a.do()

    a.finish()
    a.do()
