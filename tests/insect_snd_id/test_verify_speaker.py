import io

import pytest
from pathlib import Path
import functional as pyfn
from returns.result import Result, safe, Success, Failure
from returns.contrib.pytest import ReturnsAsserts

from pytorchstudy.insect_snd_id.verify_speaker import same_gh_test_set, diff_gh_test_set

# https://medium.com/testcult/intro-to-test-framework-pytest-5b1ce4d011ae
# https://medium.com/beyn-technology/hands-on-start-testing-with-pytest-1ef39e59176a
# https://medium.com/ideas-at-igenius/fixtures-and-parameters-testing-code-with-pytest-d8603abb390a
# https://betterprogramming.pub/understand-5-scopes-of-pytest-fixtures-1b607b5c19ed


@pytest.mark.parametrize(
    "gh, wav_list, expected",
    [
        ("gh-4", ["gh-4/1.wav"], AssertionError("list must have two or more items")),
        (
            "gh-4",
            ["gh-4/1.wav", "gh-4/2.wav", "gh-4/3.wav"],
            [
                (True, ("gh-4", "gh-4/1.wav"), ("gh-4", "gh-4/2.wav")),
                (True, ("gh-4", "gh-4/1.wav"), ("gh-4", "gh-4/3.wav")),
                (True, ("gh-4", "gh-4/2.wav"), ("gh-4", "gh-4/3.wav")),
            ],
        ),
        (
            "gh-4",
            ["gh-4/1.wav", "gh-4/2.wav", "gh-4/3.wav", "gh-4/4.wav"],
            [
                (True, ("gh-4", "gh-4/1.wav"), ("gh-4", "gh-4/2.wav")),
                (True, ("gh-4", "gh-4/1.wav"), ("gh-4", "gh-4/3.wav")),
                (True, ("gh-4", "gh-4/1.wav"), ("gh-4", "gh-4/4.wav")),
                (True, ("gh-4", "gh-4/2.wav"), ("gh-4", "gh-4/3.wav")),
                (True, ("gh-4", "gh-4/2.wav"), ("gh-4", "gh-4/4.wav")),
                (True, ("gh-4", "gh-4/3.wav"), ("gh-4", "gh-4/4.wav")),
            ],
        ),
    ],
)
def test_same_gh_test_set(returns: ReturnsAsserts, gh, wav_list, expected):
    match expected:
        case AssertionError():
            with pytest.raises(AssertionError, match=str(expected)):
                ret = same_gh_test_set(gh, wav_list)
                print(ret)
        case list():
            ret = same_gh_test_set(gh, wav_list)
            print(ret)
            assert ret == expected

    #     case Success():
    #         returns.assert_equal(ret, expected)
    #     case Failure():
    #         returns.assert_equal(ret.alt(str), expected.alt(str))

    # returns.assert_equal(
    #     Failure(Exception("1")).alt(str), Failure(Exception("1")).alt(str)
    # )
    # returns.assert_equal(Failure("1"), Failure("1"))


@pytest.mark.parametrize(
    "gh, wav_dict, expected",
    [
        (
            "gh-4",
            {"gh-4": ["gh-4/1.wav"]},
            AssertionError("gh-4 list must have two or more items"),
        ),
        (
            "gh-4",
            {"gh-4": ["gh-4/1.wav", "gh-4/2.wav"], "gh-1": ["gh-1/1.wav"]},
            AssertionError("gh-1 list must have two or more items"),
        ),
        (
            "gh-4",
            {
                "gh-4": ["gh-4/1.wav", "gh-4/2.wav", "gh-4/3.wav"],
                "gh-1": ["gh-1/1.wav", "gh-1/2.wav"],
            },
            [
                (False, ("gh-4", "gh-4/1.wav"), ("gh-1", "gh-1/2.wav")),
                (False, ("gh-4", "gh-4/2.wav"), ("gh-1", "gh-1/1.wav")),
                (False, ("gh-4", "gh-4/3.wav"), ("gh-1", "gh-1/2.wav")),
            ],
        ),
        # (
        #     "gh-4",
        #     ["gh-4/1.wav", "gh-4/2.wav", "gh-4/3.wav", "gh-4/4.wav"],
        #     [
        #         (True, ("gh-4", "gh-4/1.wav"), ("gh-4", "gh-4/2.wav")),
        #         (True, ("gh-4", "gh-4/1.wav"), ("gh-4", "gh-4/3.wav")),
        #         (True, ("gh-4", "gh-4/1.wav"), ("gh-4", "gh-4/4.wav")),
        #         (True, ("gh-4", "gh-4/2.wav"), ("gh-4", "gh-4/3.wav")),
        #         (True, ("gh-4", "gh-4/2.wav"), ("gh-4", "gh-4/4.wav")),
        #         (True, ("gh-4", "gh-4/3.wav"), ("gh-4", "gh-4/4.wav")),
        #     ],
        # ),
    ],
)
def test_diff_gh_test_set(returns: ReturnsAsserts, gh, wav_dict, expected):
    def right_match(t: tuple) -> bool:
        gen_p, exp_p = t
        if gen_p[0] != exp_p[0]:
            return False

        if gen_p[1] != exp_p[1]:
            return False

        if gen_p[2][0] != exp_p[2][0]:
            return False

        return True

    match expected:
        case AssertionError():
            with pytest.raises(AssertionError, match=str(expected)):
                ret = diff_gh_test_set(gh, wav_dict)
                print(ret)
        case list():
            ret = diff_gh_test_set(gh, wav_dict)
            print(ret)
            assert pyfn.seq(ret).zip(pyfn.seq(expected)).for_all(right_match)
