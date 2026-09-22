# -*- coding: utf-8 -*-
import math

import services.rotation_math as rm


def test_identity_gives_zero():
    r = rm.relative_rotation([1, 0, 0, 0], [1, 0, 0, 0])
    assert r.theta_deg == 0.0
    assert r.axis == (0.0, 0.0, 0.0)


def test_ninety_degrees_about_z():
    q2 = rm.axis_angle_to_quaternion((0, 0, 1), 90.0)
    r = rm.relative_rotation([1, 0, 0, 0], q2)
    assert abs(r.theta_deg - 90.0) < 1e-6
    assert abs(r.axis[2] - 1.0) < 1e-6


def test_one_eighty_about_x():
    q2 = rm.axis_angle_to_quaternion((1, 0, 0), 180.0)
    r = rm.relative_rotation([1, 0, 0, 0], q2)
    assert abs(r.theta_deg - 180.0) < 1e-6


def test_arbitrary_axis_60_degrees():
    axis = (1 / math.sqrt(3),) * 3
    q2 = rm.axis_angle_to_quaternion(axis, 60.0)
    r = rm.relative_rotation([1, 0, 0, 0], q2)
    assert abs(r.theta_deg - 60.0) < 1e-6
    for a, b in zip(r.axis, axis):
        assert abs(a - b) < 1e-6


def test_double_cover_same_theta():
    q2 = rm.axis_angle_to_quaternion((0, 0, 1), 120.0)
    neg = [-x for x in q2]
    r1 = rm.relative_rotation([1, 0, 0, 0], q2)
    r2 = rm.relative_rotation([1, 0, 0, 0], neg)
    assert abs(r1.theta_deg - r2.theta_deg) < 1e-6


def test_wraparound_is_continuous():
    q179 = rm.axis_angle_to_quaternion((0, 0, 1), 179.0)
    qn179 = rm.axis_angle_to_quaternion((0, 0, 1), -179.0)
    r = rm.relative_rotation(q179, qn179)
    assert abs(r.theta_deg - 2.0) < 1e-6


def test_euler_quat_roundtrip():
    for yaw, pitch, roll in [(0, 0, 0), (30, 10, -20), (-120, 45, 90)]:
        q = rm.euler_to_quaternion(yaw, pitch, roll)
        ry, rp, rr = rm.quaternion_to_euler(q)
        assert abs(ry - yaw) < 1e-6
        assert abs(rp - pitch) < 1e-6
        assert abs(rr - roll) < 1e-6


def test_unnormalized_quaternions_are_normalized():
    r = rm.relative_rotation([2, 0, 0, 0], rm.axis_angle_to_quaternion((0, 1, 0), 45.0))
    assert abs(r.theta_deg - 45.0) < 1e-6
