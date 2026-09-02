"""
Unit tests for the counting geometry in footfall.tracker: the free
functions (_side_of_line, _point_in_polygon) and the stateful methods
(line crossing, zone dwell, geometry rescale, the box sanity filter).

A FootfallTracker is built with model=object() so no YOLO weights load.
"""

import pytest

from footfall.tracker import FootfallTracker, Point, _point_in_polygon, _side_of_line


class RecordingSink:
    run_id = "test-run"

    def __init__(self):
        self.events = []

    def emit(self, event, track_id, value=None):
        self.events.append((event, track_id, value))

    def close(self):
        pass


def _tracker(**kw):
    kw.setdefault("source", "unused")
    kw.setdefault("model", object())
    kw.setdefault("event_sink", RecordingSink())
    return FootfallTracker(**kw)


# -- _side_of_line ---------------------------------------------------


def test_side_of_line_sign_and_zero():
    a, b = Point(0, 0), Point(0, 10)          # vertical, pointing +y
    assert _side_of_line(Point(5, 5), a, b) < 0     # right of a->b
    assert _side_of_line(Point(-5, 5), a, b) > 0    # left of a->b
    assert _side_of_line(Point(0, 5), a, b) == 0    # on the line
    assert _side_of_line(Point(0, 99), a, b) == 0   # on the infinite extension


# -- _point_in_polygon ------------------------------------------


@pytest.fixture
def square():
    return [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]


def test_point_in_polygon_inside_outside_edge(square):
    assert _point_in_polygon(Point(5, 5), square) is True
    assert _point_in_polygon(Point(20, 20), square) is False
    assert _point_in_polygon(Point(-1, 5), square) is False
    assert _point_in_polygon(Point(0, 5), square) is True    # edge counts as in


# -- line crossing state machine -----------------------------


def test_first_observation_does_not_count():
    t = _tracker(line=(Point(0, 0), Point(0, 10)))
    t._update_line_crossing(1, Point(5, 5))
    assert (t.count_in, t.count_out) == (0, 0)
    assert t._sink.events == []


def test_crossing_counts_in_then_out():
    t = _tracker(line=(Point(0, 0), Point(0, 10)))
    t._update_line_crossing(1, Point(5, 5))     # side < 0, first
    t._update_line_crossing(1, Point(-5, 5))    # -> side >= 0 : IN
    assert (t.count_in, t.count_out) == (1, 0)
    t._update_line_crossing(1, Point(5, 5))     # -> side < 0 : OUT
    assert (t.count_in, t.count_out) == (1, 1)
    assert [e[0] for e in t._sink.events] == ["line_in", "line_out"]
    assert sum(t._in_by_minute.values()) == 1


def test_staying_on_one_side_never_counts():
    t = _tracker(line=(Point(0, 0), Point(0, 10)))
    for _ in range(5):
        t._update_line_crossing(1, Point(5, 5))
    assert (t.count_in, t.count_out) == (0, 0)


def test_two_tracks_are_counted_independently():
    t = _tracker(line=(Point(0, 0), Point(0, 10)))
    for tid in (1, 2):
        t._update_line_crossing(tid, Point(5, 5))
        t._update_line_crossing(tid, Point(-5, 5))
    assert t.count_in == 2


def test_no_line_is_a_no_op():
    t = _tracker(line=None)
    t._update_line_crossing(1, Point(5, 5))
    t._update_line_crossing(1, Point(-5, 5))
    assert (t.count_in, t.count_out) == (0, 0)


# -- zone dwell ------------------------------------------------


def test_zone_enter_exit_records_one_dwell():
    zone = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
    t = _tracker(zone=zone)

    t._update_zone_dwell(1, Point(5, 5), now=100.0)      # enter
    assert t._current_queue_length() == 1
    assert t._dwell_records == []

    t._update_zone_dwell(1, Point(5, 5), now=105.0)      # still inside
    assert t._current_queue_length() == 1

    t._update_zone_dwell(1, Point(50, 50), now=108.5)    # exit
    assert t._current_queue_length() == 0
    assert t._dwell_records == [(1, 8.5)]
    assert [e[0] for e in t._sink.events] == ["zone_enter", "zone_exit"]
    assert t._sink.events[-1][2] == "8.5s"


def test_never_entering_the_zone_records_nothing():
    zone = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
    t = _tracker(zone=zone)
    t._update_zone_dwell(1, Point(99, 99), now=1.0)
    assert t._dwell_records == []
    assert t._sink.events == []


# -- _fit_geometry -------------------------------------------


def test_fit_geometry_no_authored_size_leaves_geometry_alone():
    line = (Point(10, 10), Point(90, 90))
    t = _tracker(line=line, geometry_size=None)
    t._fit_geometry(200, 400)
    assert t._geometry_fitted is True
    assert t.line == line


def test_fit_geometry_same_size_is_a_no_op():
    line = (Point(10, 10), Point(90, 90))
    t = _tracker(line=line, geometry_size=(100, 100))
    t._fit_geometry(100, 100)
    assert t.line == line


def test_fit_geometry_rescales_line_zone_and_ignore():
    t = _tracker(
        line=(Point(10, 10), Point(90, 90)),
        zone=[Point(0, 0), Point(50, 0), Point(50, 50), Point(0, 50)],
        ignore_zones=[[Point(0, 0), Point(10, 0), Point(10, 10)]],
        geometry_size=(100, 100),
    )
    t._fit_geometry(200, 400)                    # sx=2, sy=4

    assert [(p.x, p.y) for p in t.line] == [(20, 40), (180, 360)]
    assert [(p.x, p.y) for p in t.zone] == [(0, 0), (100, 0), (100, 200), (0, 200)]
    assert [(p.x, p.y) for p in t.ignore_zones[0]] == [(0, 0), (20, 0), (20, 40)]


# -- box sanity filter + helpers -----------------------------


def test_plausible_person_rejects_degenerate_short_and_wide():
    t = _tracker(min_box_height=30, max_aspect=2.0)
    assert t._plausible_person((0, 0, 10, 0)) is False       # zero height
    assert t._plausible_person((0, 0, 0, 20)) is False       # zero width
    assert t._plausible_person((0, 0, 10, 20)) is False      # 20 < min height
    assert t._plausible_person((0, 0, 60, 20)) is False      # aspect 3.0 > 2.0
    assert t._plausible_person((0, 0, 10, 40)) is True


def test_centroid_and_ignored_region():
    ignore = [[Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]]
    t = _tracker(ignore_zones=ignore)
    assert (t._centroid((0, 0, 10, 20)).x, t._centroid((0, 0, 10, 20)).y) == (5, 10)
    assert t._in_ignored_region(Point(5, 5)) is True
    assert t._in_ignored_region(Point(50, 50)) is False
