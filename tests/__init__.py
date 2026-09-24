"""Test suite for the black & white candidacy framework.

Run with either runner:

    python -m unittest discover -v
    python -m pytest -v

Every test name reads as Given / When / Then, and the body repeats the three
clauses as comments so the expectation is legible without reading assertions.
The suite is written against stubs that raise NotImplementedError, so a clean
checkout is expected to be entirely red. That is the starting state, not a bug.
"""
