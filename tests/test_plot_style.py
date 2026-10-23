"""Every series has to be told apart from every other one, on paper and in greyscale."""

from __future__ import annotations

import pytest

from scenport.plots import style

ORDERS = [
    pytest.param(style.FORECASTER_ORDER, id="forecasters"),
    pytest.param(style.ALLOCATOR_ORDER, id="allocators"),
]


def _signature(name: str, order: tuple[str, ...]) -> tuple:
    s = style.style_for(name, order)
    return (s["color"], str(s["linestyle"]), s["marker"])


@pytest.mark.parametrize("order", ORDERS)
def test_no_two_series_look_the_same(order):
    """Colour, line style and marker cycle at coprime periods, so nothing repeats.

    With all three channels on a period of six, the first and seventh series would
    be identical in every respect. In FORECASTER_ORDER those are the
    historical-simulation baseline and Prophet tuned on RMSE, a pair the paper asks
    the reader to compare.
    """
    seen: dict[tuple, str] = {}
    for name in order:
        if name in style.BACKDROP:
            continue
        sig = _signature(name, order)
        assert sig not in seen, (
            f"{name} and {seen[sig]} are drawn identically: {sig}"
        )
        seen[sig] = name


@pytest.mark.parametrize("order", ORDERS)
def test_greyscale_still_separates(order):
    """Drop the colour and the line style plus marker must still be unique."""
    seen: dict[tuple, str] = {}
    for name in order:
        if name in style.BACKDROP:
            continue
        s = style.style_for(name, order)
        sig = (str(s["linestyle"]), s["marker"])
        assert sig not in seen, (
            f"{name} and {seen[sig]} collapse to the same thing in print: {sig}"
        )
        seen[sig] = name


@pytest.mark.parametrize("order", ORDERS)
def test_backdrop_series_are_achromatic(order):
    for name in style.BACKDROP:
        if name not in order:
            continue
        assert style.style_for(name, order)["color"] == style.REFERENCE


@pytest.mark.parametrize("order", ORDERS)
def test_backdrop_series_differ_from_each_other(order):
    """The NAV figure draws 1/N and the cap-weighted index side by side.

    Both are achromatic on purpose, so the dash pattern is the only thing left to
    tell them apart. Without it the legend shows two identical grey swatches.
    """
    seen: dict[str, str] = {}
    for name in style.BACKDROP:
        if name not in order:
            continue
        dashes = str(style.style_for(name, order)["linestyle"])
        assert dashes not in seen, f"{name} and {seen[dashes]} share a dash pattern"
        seen[dashes] = name


def test_channel_periods_are_coprime():
    """The property the other tests rely on, asserted directly."""
    from math import gcd

    n_colour = len(style.PALETTE)
    n_line = len(style.LINESTYLES)
    n_marker = len(style.MARKERS)
    assert gcd(n_colour, n_line) == 1
    assert gcd(n_colour, n_marker) == 1
    assert gcd(n_line, n_marker) == 1

    longest = max(len(style.FORECASTER_ORDER), len(style.ALLOCATOR_ORDER))
    assert n_colour * n_line >= longest


def test_every_series_has_a_readable_label():
    for name in style.FORECASTER_ORDER + style.ALLOCATOR_ORDER:
        assert name in style.PRETTY, f"{name} would show up as a raw key"
        assert style.label(name).strip()
