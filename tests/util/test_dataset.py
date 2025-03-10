import io

import pytest
from pathlib import Path
from hypothesis import given, strategies as st

from pytorchstudy.util.dataset import random_pick_except_me

# https://medium.com/testcult/intro-to-test-framework-pytest-5b1ce4d011ae
# https://medium.com/beyn-technology/hands-on-start-testing-with-pytest-1ef39e59176a
# https://medium.com/ideas-at-igenius/fixtures-and-parameters-testing-code-with-pytest-d8603abb390a
# https://betterprogramming.pub/understand-5-scopes-of-pytest-fixtures-1b607b5c19ed
# property test
# https://www.freecodecamp.org/news/intro-to-property-based-testing-in-python-6321e0c2f8b/


@given(
    st.lists(st.integers(), min_size=100, max_size=100),
    st.integers(min_value=0, max_value=99),
)
def test_random_pick_except_me(fl, except_idx):
    ret = random_pick_except_me(fl, except_idx)
    assert ret is not None
    _, idx = ret
    assert idx != except_idx
