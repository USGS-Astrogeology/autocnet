from collections import deque
import unittest

from .. import observable


class TestObservableABC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        class Tester(observable.Observable):

            def __init__(self):
                self.a = 1
                self.b = 2
                self.attrs = ['a', 'b']

                self._action_stack = deque(maxlen=10)
                self._current_action_stack = 0
                self._observers = set()

                state_package = {'a': self.a,
                                 'b': self.b}

                self._action_stack.append(state_package)

            def foo(self):
                self.a += 1
                self.b += 1

                state_package = {'a': self.a,
                                 'b': self.b}

                self._action_stack.append(state_package)
                self._current_action_stack = len(self._action_stack) - 1
                self._notify_subscribers()

        cls.TestClass = Tester()

    def test_do_undo(self):

        self.assertEqual(self.TestClass.a, 1)
        self.assertEqual(self.TestClass.b, 2)

        self.TestClass.foo()

        self.assertEqual(self.TestClass.a, 2)
        self.assertEqual(self.TestClass.b, 3)

        self.TestClass.rollback()

        self.assertEqual(self.TestClass.a, 1)
        self.assertEqual(self.TestClass.b, 2)

        self.TestClass.rollforward()

        self.assertEqual(self.TestClass.a, 2)
        self.assertEqual(self.TestClass.b, 3)

    def test_subscribe(self):
        self.alertvalue = 0

        def alert():
            self.alertvalue += 1

        self.TestClass.subscribe(alert)
        self.assertEqual(self.alertvalue, 0)
        self.TestClass.foo()
        self.assertEqual(self.alertvalue, 1)
