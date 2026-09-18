"""
v2/ttc.py — the one Time-to-Collision formula, used identically for
every track regardless of which sensor(s) produced it, and for
ground-truth instances in evaluate.py (same function, same formula, so
the comparison is genuinely apples-to-apples).

Sign convention: R10 in v2_architecture_brief.md specifies closing
speed as the radial component of "the track's velocity" toward the ego
vehicle, using only the track's own velocity. That literal formula was
checked against the brief's author and confirmed to be missing the
ego-velocity term deliberately omitted in the summary, not by design --
both the tracked object and the ego vehicle can be moving, so true
closing speed needs RELATIVE velocity (this is also how the old
pipeline's compute_ttc() worked). The extended formula used here:

    (dx, dy)         = (ego_x - track_x, ego_y - track_y)
    distance         = hypot(dx, dy)
    (rel_vx, rel_vy) = (track_vx - ego_vx, track_vy - ego_vy)
    closing_speed    = (rel_vx*dx + rel_vy*dy) / distance

Positive closing_speed means the track is closing on the ego vehicle
(radial component of relative velocity along the line of sight, from
track toward ego). If closing_speed <= 0, ttc is float('inf') -- never
negative.

"Never a raw division artifact" (R10) needs one more guard beyond
closing_speed <= 0, found by running this on the real dataset: a
closing speed that is technically positive but negligibly small (very
common for an object moving roughly tangentially to the ego vehicle at
any given instant) still divides distance by that tiny number,
producing an enormous but FINITE ttc (observed: up to ~190 MILLION
seconds) that carries no real collision-relevant meaning -- exactly
the division artifact R10 already says to avoid, just not anticipated
by the literal closing_speed <= 0 boundary alone. config.MIN_CLOSING_SPEED
closes that gap: closing_speed <= MIN_CLOSING_SPEED (not just <= 0) now
returns inf. Deadbanding the unstable division at the SOURCE (here) was
chosen over capping the output ttc afterward in evaluate.py, since a
post-hoc ttc cap can't distinguish "large because closing speed is
genuinely near-zero" (broken) from "large but legitimate, because the
object is far away and closing slowly" (not broken) -- only
closing_speed itself carries that distinction.

MIN_CLOSING_SPEED is NOT the same thing as kalman_track.py's/config's
MIN_TRUSTED_SPEED, and the two are not redundant: MIN_TRUSTED_SPEED
gates a track's raw velocity MAGNITUDE (is this object moving at all,
vs. detector/clustering jitter on something stationary).
MIN_CLOSING_SPEED gates the CLOSING/RADIAL-TO-EGO COMPONENT
specifically -- an object can have plenty of raw speed, well above
MIN_TRUSTED_SPEED, while still having near-zero closing speed if its
motion is roughly tangential to the ego vehicle's line of sight.

Because evaluate.compute_ground_truth_ttc() calls this SAME function,
ground-truth TTC gets the identical deadband automatically -- no
second filter to keep in sync by hand, and predicted vs. ground-truth
TTC stay symmetric.

Always call this with a track's own KF velocity state (Track.velocity
or Track.trusted_velocity()) -- never a raw sensor-reported velocity
field such as radar Doppler (R10).
"""
import math

from v2 import config


def compute_ttc(track_x, track_y, track_vx, track_vy, ego_x, ego_y, ego_vx, ego_vy,
                 min_closing_speed=None):
    """
    min_closing_speed: overridable for tests; defaults to
        config.MIN_CLOSING_SPEED.

    Returns (distance, closing_speed, ttc):
        distance: metres between track and ego.
        closing_speed: m/s, positive when the track is closing on ego
            (radial component of relative velocity toward ego).
        ttc: seconds. float('inf') if not closing enough to be a
            meaningful collision estimate (closing_speed <=
            min_closing_speed -- see module docstring for why this is
            not just closing_speed <= 0) or if distance is exactly zero
            (coincident -- no meaningful radial direction to measure a
            closing speed along).
    """
    min_closing_speed = config.MIN_CLOSING_SPEED if min_closing_speed is None else min_closing_speed

    dx = ego_x - track_x
    dy = ego_y - track_y
    distance = math.hypot(dx, dy)

    if distance <= 0.0:
        return distance, 0.0, float("inf")

    rel_vx = track_vx - ego_vx
    rel_vy = track_vy - ego_vy
    closing_speed = (rel_vx * dx + rel_vy * dy) / distance

    if closing_speed <= min_closing_speed:
        return distance, closing_speed, float("inf")

    ttc = distance / closing_speed
    return distance, closing_speed, ttc
