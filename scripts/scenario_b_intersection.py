import carla
import math
import csv
import os
import random
import datetime
import statistics

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOST              = 'localhost'
PORT              = 2000
TIMEOUT           = 30.0

# ── PILOT MODE ──────────────────────────────────────────────────────────────
# This is a full rewrite (2026-07-09) of a script previously known to be
# broken: manual steer_toward_waypoint() navigation crashed repeatedly and
# 69% of logged "yield" states weren't even at a real junction.
#
# Six pilots later (2026-07-11), yield rate stabilised at 80-87.5% with every
# failure category resulting in a discarded attempt, never a logged bad pair
# (yield_verify_fails = 0 for three pilots straight). Decision: stop
# iterating, proceed to the full run. The old contaminated
# scenario_b_intersection.csv (1875 rows, 69% is_junction_t=False) was
# archived to scenario_b_intersection_ORIGINAL_CONTAMINATED.csv before this
# flip — do not treat that file as valid data going forward.
#
# Reset to True 2026-07-11 (again) after the full 500-episode run: that run's
# scenario_b_intersection.csv (1461 pairs) is clean in every way that was
# checked at the time (100% is_junction_t, 100% outcome=safe) but has ZERO
# decision-T variance — speed_t is identically 0.0000 across all 1461 pairs,
# because the yield hold always fully brakes. That means B cannot support
# the primary correlation ablation as collected; it can only feed the
# demoted secondary D-mean comparison. See YIELD_SPEED_RANGE below for the
# fix (sampled rolling-yield speed) — that file is superseded for primary
# evidence purposes, do not delete it, but pilot and rerun before relying on
# B for anything beyond the secondary comparison.
#
# Pilot verified 2026-07-12: variance gate passed cleanly (std(speed_t)=0.74,
# target-vs-realized-speed r=0.997 — the controller tracks the sampled
# target almost exactly). BUT the correlation this was ultimately for isn't
# there yet at pilot scale: r(speed_t, rho_actual)=0.22 pair-level (p=0.07),
# 0.27 episode-level (permutation p=0.26-0.28, n=19 episodes) — not
# significant. Root cause found: rho_actual is itself nearly constant here
# (std=0.008) because junction_risk is pinned at 1.0 (is_junction_t1 always
# True by design) and speed_risk is pinned at 0.2 (speed_t1 always clusters
# 5.5-5.9 m/s, under the formula's 6.0 m/s threshold) for every successfully
# logged pair — only the small lane_dev term has room to move. Decision:
# flip to False anyway and run the full 500 — 25x the data may surface a
# real-but-weak signal that 65 pairs can't. If it's STILL not significant at
# full scale, the next fix is get_risk_score()'s junction_risk/speed_risk
# terms (make them continuous instead of near-step-functions, same idea as
# scenario_c_braking.py's soft_cap), not another yield-sampling change.
#
# CONFIRMED 2026-07-12, full 500-episode run, independently verified against
# the raw CSV (not just this comment's prediction): still not significant.
# pair-level r(speed_t, rho_actual)=0.089 p=6.2e-4 (n=1476) — "significant"
# only because pooling correlated pairs inflates it (pseudo-replication, see
# Methodology status above); episode-level (the correct unit) r=0.076
# p=0.094, permutation p=0.093, n=487 episodes. rho_actual std=0.0141 full
# scale (even tighter than the pilot's 0.008 relative to n). Target-tracking
# itself is still excellent (r=0.998, sampling side confirmed fine). This is
# exactly the predicted outcome — proceeding with the junction_risk/speed_risk
# fix below, not another yield-sampling change.
#
# FIX APPLIED 2026-07-12: get_risk_score()'s junction_risk and speed_risk are
# now continuous (see estimate_junction_progress() / continuous_speed_risk()
# below) instead of near-step-functions. junction_risk was pinned at 1.0
# because is_junction_t1 is True for ~100% of logged pairs by construction
# (the cross maneuver frequently ends still inside the junction) — replaced
# with a continuous "how much further to the junction exit" measure via
# forward waypoint probing, normalized to [0,1]. speed_risk was pinned at 0.2
# because its 1.0-6.0 m/s branch was a flat shelf and CROSS_SPEED=6.0 pulls
# speed_t1 into exactly that shelf — replaced with a smooth U-shaped curve
# around a reasoned optimal crossing speed, so small real speed differences
# translate into small but nonzero risk differences instead of being
# collapsed to the same constant. Weights (0.5/0.3/0.2) deliberately left
# unchanged — the fix targets continuity, not reweighting, so any resulting
# correlation can't be dismissed as tuned-after-the-fact. Locked in before
# rerunning, same discipline as scenario_c_braking.py's a priori severity
# metric. PILOT_MODE forced back to True — this is an unvalidated formula
# change and must be piloted (15-20 episodes) before any full rerun, per
# house style.
#
# PILOT CONFIRMED 2026-07-12, independently verified against the raw pilot
# CSV (bash mount stale again, same recurring issue — Read tool showed the
# true, complete 66-row/19-episode file, matching the console output
# exactly): rho_actual std jumped 0.0141 -> 0.1133 (8x), range 0.214-0.591 —
# the pinning is broken, gate 1 passes cleanly. Gate 2 (correlation
# significance) is NOT yet clear at this scale: pair-level r=-0.252 p=0.041
# (n=66), episode-level r=-0.367 p=0.123, permutation p=0.126 (n=19
# episodes) — not significant, and notably the correlation is NEGATIVE, not
# positive. Working theory: a faster yield carries more momentum into the
# cross, so by the time rho_actual is sampled post-cross a fast-yielding
# vehicle has typically pushed further through the junction than a slow one
# — less remaining exposure at that snapshot, not more. That's a coherent
# story, not obviously a bug, but it means if the full run comes back
# significant, expect the sign to be negative (faster yield -> lower
# carried-forward risk), not a confirmation of the original "riskier
# decision T propagates to riskier T+1" direction. Decision: proceed to the
# full 500 anyway — the blocking gate (variance) passed, and n=19 can't
# resolve r=-0.37 either way — but interpret the eventual full-scale p-value
# together with its sign, not in isolation.
# ── INSTRUMENTATION PASS (2026-07-12, after the third Fable audit) ──────────
# The 500-episode run above is BANKED as primary evidence (episode-level
# r=-0.150, p=9.4e-4, robust to junction fixed effects and to dropping the
# 6 most extreme junctions). But the audit falsified the "momentum"
# mechanism story: displacement carries ~none of the effect (within-junction
# corr(disp, residual)=-0.05), and per-junction r(speed_t, rho) ranges from
# -0.83 to +0.81 — a single pooled mechanism can't produce sign flips by
# junction. The mechanism can't be decomposed further because only the
# combined rho_actual was ever logged. This pass adds INSTRUMENTATION ONLY:
#   1. junction_progress / speed_risk / lane_dev logged as separate CSV
#      columns at both T and T+1 (plus rho_t, which was computed at the
#      yield every run but never written anywhere).
#   2. Every discarded attempt logged to a *_discards.csv sidecar with the
#      sampled yield-speed target (when one had been sampled yet) and
#      location — closes the selection-bias question by evidence.
#   3. Per-episode failure counters persisted to *_episode_stats.csv — so
#      the transit_collisions anomaly (5,839 in the full run vs ~2,050
#      linear-scaled from the pilot) is auditable from disk next session.
# NO formula term, weight, range, or control constant is touched — the risk
# math must stay byte-identical to the validated run so the instrumented
# rerun is comparable to the banked result.
# Output goes to scenario_b_instrumented*.csv, NOT scenario_b_intersection.csv:
# the banked primary CSV must never be overwritten by a mechanism run.
# Full scale for this pass is ~100 episodes — mechanism decomposition needs
# component-level variance, not another 500-episode headline sample.
PILOT_MODE        = False
NUM_EPISODES      = 20 if PILOT_MODE else 100
# ──────────────────────────────────────────────────────────────────────────────

PAIRS_PER_EPISODE = 4
TICKS_YIELD       = 30    # ticks to hold at intersection entry (decision T)
TICKS_CROSS       = 50    # ticks to cross through intersection (decision T+1)

# ── Why this rewrite uses autopilot for navigation ───────────────────────────
# The original script drove the ENTIRE episode — approach, yield, cross,
# clear — with a hand-written steer_toward_waypoint() controller. That
# controller was too simple for real intersection geometry: it crashed
# repeatedly, kept targeting the same 5-6 junctions, and 69% of "yield"
# states weren't actually at a junction. Per Approach B (readme.md), the
# ONLY things that need to be ground-truth scripted decisions are the yield
# and the cross — navigating to the junction is transit, not a decision, so
# using CARLA's own autopilot (Traffic Manager) for that transit does not
# contaminate the RPI measurement. Traffic Manager already knows how to
# follow real road geometry through turns, which our hand-written controller
# never reliably did. Manual control is taken over ONLY for the yield-hold
# and the cross maneuver, then handed back to autopilot afterward.
# MAX_NAV_TICKS was 300 (15 sim-seconds) — a 2026-07-11 pilot with the
# approach-phase fix in place showed yield_verify_fails drop to 0 (that fix
# worked) but nav_timeouts rise to 18/20 episodes, now the dominant failure
# mode. Pattern: most episodes get exactly 1 successful pair then time out
# searching for the SECOND junction (i.e. 15s often isn't enough to reach
# the next junction after already using up a nearby one); a few spawn points
# (e.g. (-56.9,140.5) / (-54.3,137.1)) got 0 pairs, timing out even on the
# first attempt — likely genuine dead zones in this map far from any
# junction. Doubled the budget to give autopilot more room/distance.
JUNCTION_LOOKAHEAD = 8.0    # metres - detect an upcoming junction while still approaching it
MAX_NAV_TICKS      = 600    # safety cap: give up looking for a junction after this many ticks
MAX_APPROACH_TICKS = 60     # safety cap: close the last ~8m under manual control (see approach_until_at_junction)
MAX_EPISODE_TICKS  = 3000   # safety cap so one bad episode can't run forever
CROSS_SPEED        = 6.0    # m/s target speed while crossing
CROSS_LOOKAHEAD    = 6.0    # metres ahead to steer toward while crossing

# Stuck detection (added 2026-07-11) — see navigate_to_next_junction() for
# the full story: a pilot showed 296-498 transit_collisions in specific
# episodes, correlating with nav_timeouts and under-logged pairs. Bail out
# early instead of burning the full MAX_NAV_TICKS budget on a vehicle wedged
# against a static prop.
STUCK_SPEED_THRESHOLD = 0.3   # m/s - below this counts as "not really moving"
STUCK_TICKS_THRESHOLD = 40    # consecutive ticks below threshold before declaring stuck (2 sim-seconds)

# ── Decision T now has real per-episode variance (added 2026-07-11) ─────────
# The yield hold previously always fully braked (brake=1.0 the whole time),
# so decision T's state was physically identical across every single pair —
# speed_t == 0.0000 for all 1461 pairs in the full run. That made it
# impossible to test whether the SEVERITY/nature of the yield (full stop vs.
# rolling creep) predicts risk at T+1, mirroring the exact defect diagnosed
# in scenario_c_braking.py's original fixed BRAKE_INTENSITY/TICKS_BRAKE.
# Sampling a per-pair target yield speed fixes this: 0.0 reproduces a full
# stop, up to YIELD_SPEED_RANGE[1] produces a "rolling yield" (slows for the
# junction without fully stopping) — see hold_position_at_speed(). This is a
# state-variance fix, not a timing one: TICKS_YIELD stays fixed, because
# varying only the hold DURATION would leave decision T's state identical
# (a car stopped 20 ticks vs. 40 ticks is in the same state either way) —
# the variance has to live in the target speed, not the clock.
YIELD_SPEED_RANGE = (0.0, 3.0)   # m/s, uniform range sampled per yield event
RNG_SEED          = 42           # reproducible yield-speed sampling

OUTPUT_DIR        = r'C:\Users\jayad\Desktop\RPI_AINN\data'
OUTPUT_FILE       = os.path.join(
    OUTPUT_DIR,
    'scenario_b_instrumented_pilot.csv' if PILOT_MODE else 'scenario_b_instrumented.csv'
)
SCENARIO_TYPE     = 'B_intersection'
WEATHER           = 'clear'
NPC_DENSITY       = 0
ERROR_LOG_FILE    = OUTPUT_FILE.replace('.csv', '_errors.log')
# Sidecar outputs (instrumentation pass, 2026-07-12) — same basename as the
# main CSV so pilot/full runs keep their sidecars paired automatically:
DISCARD_LOG_FILE   = OUTPUT_FILE.replace('.csv', '_discards.csv')
EPISODE_STATS_FILE = OUTPUT_FILE.replace('.csv', '_episode_stats.csv')
# ──────────────────────────────────────────────────────────────────────────────


def log_error(msg):
    """Append a timestamped error to a persistent log file — terminal
    scrollback is unreliable for long unattended runs (see scenario_c_braking.py
    for why this was added: it's how 26/400 episodes' failure reasons got lost
    in the 2026-07-09 full run)."""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")


# ── Continuous risk terms (added 2026-07-12) ────────────────────────────────
# Both replace hard step-functions that were confirmed (full 500-episode run,
# see PILOT_MODE comment block above) to pin rho_actual to a near-constant
# value: junction_risk was always 1.0 (is_junction_t1 True ~100% of the time
# by construction) and speed_risk was always 0.2 (speed_t1 clustered in the
# old formula's flat 1.0-6.0 m/s shelf, driven straight there by
# CROSS_SPEED=6.0). Neither fix changes the get_risk_score() weights
# (0.5/0.3/0.2) — only how each term is computed — so this isolates the
# continuity fix from any reweighting.

JUNCTION_PROBE_MAX   = 25.0   # metres - cap on how far forward we probe to find the junction exit
JUNCTION_PROBE_STEP  = 1.0    # metres - probing resolution
OPTIMAL_CROSS_SPEED  = 4.0    # m/s - reasoned "safest" crossing speed: fast enough not to linger
                               # exposed in the junction, slow enough to react to cross traffic;
                               # deliberately NOT set equal to CROSS_SPEED=6.0 (a control target,
                               # not a claim about what's safest) to avoid the new formula being
                               # trivially minimized by the controller's own aim point
SPEED_RISK_SCALE     = 6.0    # m/s - controls how quickly speed_risk rises away from optimal


def estimate_junction_progress(waypoint):
    """
    Continuous replacement for the old is_junction boolean. Returns a value
    in [0,1]: high when the vehicle has just entered a junction (or is well
    inside it, far from the exit), falling smoothly toward 0 as it
    approaches the junction exit, and a small residual value when not yet
    in a junction at all (short falloff based on how soon one is coming up,
    replacing the old flat 0.6/0.1 approach branch).

    Walks forward from the current waypoint in JUNCTION_PROBE_STEP
    increments (via waypoint.next(d)) until it finds a point that is no
    longer is_junction, and normalizes that remaining distance against
    JUNCTION_PROBE_MAX. This directly targets the confirmed defect: at
    T+1 (post-cross), is_junction was True for ~100% of logged pairs, so
    the old boolean carried zero information about how much junction
    exposure was actually left — this does.
    """
    if waypoint is None:
        return 1.0

    if not waypoint.is_junction:
        # Not in a junction yet — soft falloff based on proximity to one,
        # scaled down (max 0.6) so it never outweighs actually being inside.
        for d in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
            ahead = waypoint.next(d)
            if ahead and ahead[0].is_junction:
                return max(0.0, 1.0 - d / 12.0) * 0.6
        return 0.05

    d = JUNCTION_PROBE_STEP
    while d <= JUNCTION_PROBE_MAX:
        ahead = waypoint.next(d)
        if not ahead or not ahead[0].is_junction:
            break
        d += JUNCTION_PROBE_STEP

    remaining = min(d, JUNCTION_PROBE_MAX)
    return min(1.0, remaining / JUNCTION_PROBE_MAX)


def continuous_speed_risk(speed_ms):
    """
    Continuous replacement for the old step function (<1.0 m/s -> 0.7 flat,
    1.0-6.0 m/s -> 0.2 flat, >6.0 m/s -> linear). The middle shelf was the
    confirmed defect: CROSS_SPEED=6.0 reliably put speed_t1 inside that
    flat band, so speed_risk carried zero information regardless of the
    actual speed. This is a smooth U-shaped curve around
    OPTIMAL_CROSS_SPEED, penalizing both lingering too slowly (more time
    exposed in the junction) and crossing too fast (less margin to react
    to cross traffic), with no flat region anywhere.
    """
    deviation = (speed_ms - OPTIMAL_CROSS_SPEED) / SPEED_RISK_SCALE
    return min(1.0, 0.15 + deviation ** 2)


def compute_risk_components(vehicle, world):
    """
    Compute the three risk components AND the combined score in one pass.

    Added 2026-07-12 (instrumentation pass): the third audit could not
    settle B's mechanism because only the combined rho_actual was logged —
    the offline residual analysis had to lump junction_progress and
    lane_dev together, which is exactly where the mechanism lives. Logging
    each component separately is simultaneously (a) the mechanism-
    decomposition data, and (b) a per-component pinning detector that would
    have caught the old junction_risk=1.0 / speed_risk=0.2 plateau in the
    very first pilot instead of after a full 500-episode run.

    The math is byte-identical to the validated 500-episode formula — same
    terms, same weights (0.5/0.3/0.2), same final rounding. The combined
    'rho' is computed from the UNROUNDED components, exactly as the old
    get_risk_score() did; component values are rounded only for logging.
    get_risk_score() below is now a thin wrapper over this, so no caller
    can drift out of sync with the logged components.
    """
    transform = vehicle.get_transform()
    velocity  = vehicle.get_velocity()
    speed_ms  = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    waypoint = world.get_map().get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    junction_risk = estimate_junction_progress(waypoint)
    speed_risk    = continuous_speed_risk(speed_ms)

    # Lane deviation (unchanged — this term was never pinned)
    if waypoint is None:
        lane_dev = 1.0
    else:
        wp_loc  = waypoint.transform.location
        dist    = math.sqrt(
            (transform.location.x - wp_loc.x)**2 +
            (transform.location.y - wp_loc.y)**2
        )
        lane_dev = min(dist / 3.0, 1.0)

    return {
        'junction_progress': round(junction_risk, 6),
        'speed_risk'       : round(speed_risk, 6),
        'lane_dev'         : round(lane_dev, 6),
        'rho'              : round(0.5 * junction_risk + 0.3 * speed_risk + 0.2 * lane_dev, 6),
    }


def get_risk_score(vehicle, world):
    """Combined score only — kept for compatibility; delegates to
    compute_risk_components() so the formula exists in exactly one place."""
    return compute_risk_components(vehicle, world)['rho']


def get_state(vehicle, world):
    transform = vehicle.get_transform()
    velocity  = vehicle.get_velocity()
    speed_ms  = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    waypoint  = world.get_map().get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    return {
        'x'          : round(transform.location.x, 4),
        'y'          : round(transform.location.y, 4),
        'z'          : round(transform.location.z, 4),
        'yaw'        : round(transform.rotation.yaw, 4),
        'speed_ms'   : round(speed_ms, 4),
        'lane_id'    : waypoint.lane_id     if waypoint else None,
        'road_id'    : waypoint.road_id     if waypoint else None,
        'is_junction': waypoint.is_junction if waypoint else False,
    }


def lane_follow_steer(vehicle, world, lookahead):
    """
    Continuous per-tick steering toward the lane centreline `lookahead`
    metres ahead. Same fix applied to scenario_c_braking.py after its pilot
    showed the old static/one-shot targeting approach failing on curves —
    recomputing the target every tick handles turns (including junction
    approaches and crossings) far more robustly than a pre-computed waypoint
    list ever did.
    """
    transform = vehicle.get_transform()
    waypoint  = world.get_map().get_waypoint(
        transform.location, project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        return 0.0

    ahead = waypoint.next(lookahead)
    if not ahead:
        return 0.0
    target_loc = ahead[0].transform.location

    vehicle_yaw = math.radians(transform.rotation.yaw)
    dx = target_loc.x - transform.location.x
    dy = target_loc.y - transform.location.y
    target_yaw = math.atan2(dy, dx)

    yaw_diff = target_yaw - vehicle_yaw
    while yaw_diff >  math.pi: yaw_diff -= 2 * math.pi
    while yaw_diff < -math.pi: yaw_diff += 2 * math.pi

    return max(-1.0, min(1.0, yaw_diff / math.pi * 2.0))


def tick_world(world, vehicle, spectator):
    """Advance one tick and move the spectator camera to follow the vehicle."""
    world.tick()
    loc   = vehicle.get_transform().location
    yaw   = vehicle.get_transform().rotation.yaw
    fwd_x = math.cos(math.radians(yaw))
    fwd_y = math.sin(math.radians(yaw))
    spectator.set_transform(carla.Transform(
        carla.Location(x=loc.x - 12 * fwd_x, y=loc.y - 12 * fwd_y, z=loc.z + 7),
        carla.Rotation(pitch=-25, yaw=yaw)
    ))


def navigate_to_next_junction(world, vehicle, spectator):
    """
    Let autopilot (Traffic Manager) drive until the vehicle is just about to
    enter a junction, then return 'found' so the caller can take manual
    control. This is the "transit," not a decision — see the CONFIG comment
    above for why that's valid under Approach B.

    Returns 'timeout' if MAX_NAV_TICKS elapses without finding a junction
    (dead end, roundabout the TM won't approach the way we expect, etc.).

    Returns 'stuck' if the vehicle's speed stays near zero for
    STUCK_TICKS_THRESHOLD consecutive ticks while autopilot is supposedly
    driving it forward. Added 2026-07-11: a pilot showed transit_collisions
    spiking to 296-498 in specific episodes (spawns clustered around
    (-56,140) to (5,130) — the same plaza area visually confirmed earlier
    with a lamppost/banner collision), correlating exactly with the
    episodes that timed out and under-logged pairs. That many collision
    events in one episode only makes sense if the vehicle got physically
    wedged against a static prop while autopilot kept commanding it forward
    into the same obstacle — not "still searching," genuinely stuck. This
    detects that early instead of burning the full 600-tick budget on it.
    """
    vehicle.set_autopilot(True)
    stuck_ticks = 0

    for _ in range(MAX_NAV_TICKS):
        tick_world(world, vehicle, spectator)

        velocity = vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        if speed_ms < STUCK_SPEED_THRESHOLD:
            stuck_ticks += 1
            if stuck_ticks >= STUCK_TICKS_THRESHOLD:
                return 'stuck'
        else:
            stuck_ticks = 0

        transform = vehicle.get_transform()
        waypoint  = world.get_map().get_waypoint(
            transform.location, project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            continue
        if waypoint.is_junction:
            # Already inside the junction — too late to log a clean "approach
            # then yield" pair this cycle. Let autopilot carry it through and
            # try again on the next leg.
            continue

        ahead = waypoint.next(JUNCTION_LOOKAHEAD)
        if any(w.is_junction for w in ahead):
            return 'found'   # approaching a junction — hand off to manual control

    return 'timeout'


def approach_until_at_junction(world, vehicle, spectator, max_ticks):
    """
    Drive under manual control (light throttle + lane-follow steering) from
    "a junction was detected ~8m ahead" until the vehicle's CURRENT waypoint
    is actually is_junction == True, then return 'found'.

    Added 2026-07-11 after a pilot showed yield_verify_fails (34) dominating
    nav_timeouts (14) more than 2-to-1: the previous version cut straight
    from "junction detected" to a hard brake (hold_position), so the
    vehicle's momentum only carried it partway there before stopping —
    it braked on the approach road, short of the junction boundary, which
    is exactly what the is_junction verification gate was correctly
    catching. This closes that gap by continuing to advance under control
    (not autopilot, which might not aim straight for the junction in a
    partial hand-off) until the vehicle is actually inside it.

    Stuck detection added 2026-07-11 (same day, after adding it to
    navigate_to_next_junction() fixed nav_timeouts but approach_timeouts
    rose from 2 to 6 — episode 18 alone had 3 approach_timeouts and 172
    transit_collisions in one episode, the same "wedged against a static
    prop" pattern as the navigate phase, just showing up here instead once
    that phase got fixed. Same threshold constants, same logic.

    Returns 'timeout' if max_ticks elapses without reaching the junction.
    Returns 'stuck' if speed stays near zero for STUCK_TICKS_THRESHOLD
    consecutive ticks. Caller should treat both as a failed attempt, same
    as a nav_timeout — don't log a pair.
    """
    stuck_ticks = 0
    for _ in range(max_ticks):
        transform = vehicle.get_transform()
        waypoint  = world.get_map().get_waypoint(
            transform.location, project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is not None and waypoint.is_junction:
            return 'found'

        velocity = vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        if speed_ms < STUCK_SPEED_THRESHOLD:
            stuck_ticks += 1
            if stuck_ticks >= STUCK_TICKS_THRESHOLD:
                return 'stuck'
        else:
            stuck_ticks = 0

        steer         = lane_follow_steer(vehicle, world, CROSS_LOOKAHEAD)
        ctrl          = carla.VehicleControl()
        ctrl.throttle = 0.3
        ctrl.brake    = 0.0
        ctrl.steer    = steer
        vehicle.apply_control(ctrl)
        tick_world(world, vehicle, spectator)

    return 'timeout'


def hold_position_at_speed(world, vehicle, spectator, n_ticks, target_speed):
    """
    Regulate the vehicle toward target_speed for n_ticks — the yield
    decision (T). target_speed=0.0 reproduces the old full-stop behaviour;
    a nonzero target (see YIELD_SPEED_RANGE) produces a "rolling yield" —
    the vehicle slows for the junction without coming to a complete stop.

    Replaces the old hold_position(), which always applied brake=1.0 for
    the full hold — see the YIELD_SPEED_RANGE comment above for why that
    made decision T's state identical across every pair. Proportional
    control, same pattern as cross_junction()'s speed regulation: brake
    proportionally when above target, light throttle when at/under it.
    """
    for _ in range(n_ticks):
        steer       = lane_follow_steer(vehicle, world, CROSS_LOOKAHEAD)
        velocity    = vehicle.get_velocity()
        speed_ms    = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_error = target_speed - speed_ms

        ctrl = carla.VehicleControl()
        if speed_error < 0:
            # above target — brake proportionally to how far over we are
            ctrl.throttle = 0.0
            ctrl.brake    = max(0.0, min(1.0, -speed_error * 0.5))
        else:
            # at/under target — light throttle to hold it, don't overshoot
            ctrl.throttle = max(0.0, min(0.3, speed_error * 0.3))
            ctrl.brake    = 0.0
        ctrl.steer = steer
        vehicle.apply_control(ctrl)
        tick_world(world, vehicle, spectator)


def cross_junction(world, vehicle, spectator, n_ticks):
    """Drive through the junction — the cross decision (T+1)."""
    for _ in range(n_ticks):
        steer    = lane_follow_steer(vehicle, world, CROSS_LOOKAHEAD)
        velocity = vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_error = CROSS_SPEED - speed_ms
        throttle    = max(0.0, min(0.7, 0.35 + speed_error * 0.1))

        ctrl          = carla.VehicleControl()
        ctrl.throttle = throttle
        ctrl.brake    = 0.0
        ctrl.steer    = steer
        vehicle.apply_control(ctrl)
        tick_world(world, vehicle, spectator)


def run_episode(episode_id, vehicle, world, spectator, writer, discard_writer,
                collision_state, rng, spawn_point):
    """
    One episode: up to PAIRS_PER_EPISODE yield->cross cycles at real
    junctions. Each cycle: autopilot navigates to the next junction approach,
    manual control takes the yield (T) and cross (T+1) decisions, then
    autopilot resumes. Pairs where is_junction_t doesn't verify True are
    discarded rather than logged — this is the fix for the 69% contamination
    in the original script.

    Diagnostic counters added 2026-07-09 after the pilot showed only 13/80
    possible pairs logged: nav_timeouts and yield_verify_fails distinguish
    "autopilot couldn't find the next junction in time" from "found a
    junction but the yield-hold verification failed," since the fix for each
    is different and I don't want to tune MAX_NAV_TICKS blind.

    approach_timeouts added 2026-07-11: a second pilot showed
    yield_verify_fails (34) dominating nav_timeouts (14) 2-to-1. Root cause:
    jumping straight from "junction detected 8m ahead" to a hard brake left
    the vehicle short of the junction boundary on momentum alone. Inserted
    approach_until_at_junction() between detection and the yield-hold to
    close that gap under manual control; this counter tracks when even that
    fails to reach the junction within its own tick budget.

    stuck_in_transit added 2026-07-11 (same day, third pilot): transit_
    collisions spiked to 296-498 in specific episodes, correlating with
    nav_timeouts. navigate_to_next_junction() now detects near-zero speed
    for a sustained stretch and bails out early rather than waiting the
    full MAX_NAV_TICKS budget — this counter is the "stuck" subset of what
    used to all be lumped into nav_timeouts.

    approach_stuck added 2026-07-11 (same day, fourth pilot): with
    nav_timeouts fixed, approach_timeouts rose from 2 to 6 and episode 18
    alone showed 3 approach_timeouts with 172 transit_collisions — the same
    "wedged against a static prop" pattern, just relocated to the approach
    phase now that the navigate phase is fixed. Same stuck-detection logic
    added to approach_until_at_junction(); this counter is the "stuck"
    subset of what used to all be lumped into approach_timeouts.

    Rolling yield added 2026-07-11 (see YIELD_SPEED_RANGE): decision T now
    samples a target creep speed instead of always fully braking. Watch
    yield_verify_fails in the next pilot — a rolling yield at up to 3 m/s
    for TICKS_YIELD=30 ticks (1.5s) can drift up to ~4.5m during the hold,
    and on a small junction that could carry the vehicle past the point
    is_junction still reads True. If yield_verify_fails jumps noticeably
    versus the pre-rolling-yield pilots, narrow YIELD_SPEED_RANGE rather
    than assuming it's the same class of failure as before.
    """
    pairs_logged        = 0
    nav_timeouts        = 0
    stuck_in_transit     = 0
    approach_timeouts   = 0
    approach_stuck      = 0
    yield_verify_fails  = 0
    total_ticks         = 0
    speed_t_values      = []   # for the end-of-run saturation sanity check
    rho_actual_values   = []   # for the end-of-run rho_actual variance gate (added 2026-07-12,
                                # same pattern as speed_t_values — this is the direct check on
                                # whether the junction_risk/speed_risk continuity fix actually
                                # broke the pinning, not just a proxy for it)
    component_values    = []   # per-pair T+1 component dicts, for the per-component
                                # variance gate (instrumentation pass, 2026-07-12)

    def log_discard(pair_attempt, failure_type, yield_speed_target=None):
        """
        Instrumentation (2026-07-12): every discarded attempt gets a sidecar
        row. yield_speed_target is None for failures that happen BEFORE the
        yield target is sampled (nav/approach failures) — that asymmetry is
        itself informative: pre-sampling failures cannot select on yield
        speed by construction, so the selection-bias audit only needs to
        compare the target distribution of yield_verify_fail discards
        against logged pairs. fail_x/fail_y map failures onto junction
        clusters / dead zones for the transit_collisions anomaly audit.
        """
        transform = vehicle.get_transform()
        velocity  = vehicle.get_velocity()
        discard_writer.writerow({
            'episode_id'          : episode_id,
            'pair_attempt'        : pair_attempt,
            'failure_type'        : failure_type,
            'yield_speed_target_t': yield_speed_target,
            'spawn_x'             : round(spawn_point.location.x, 4),
            'spawn_y'             : round(spawn_point.location.y, 4),
            'fail_x'              : round(transform.location.x, 4),
            'fail_y'              : round(transform.location.y, 4),
            'fail_speed'          : round(math.sqrt(
                velocity.x**2 + velocity.y**2 + velocity.z**2), 4),
        })

    for pair in range(PAIRS_PER_EPISODE):
        if total_ticks > MAX_EPISODE_TICKS:
            break

        nav_status = navigate_to_next_junction(world, vehicle, spectator)
        total_ticks += MAX_NAV_TICKS   # conservative upper bound for the cap check
        if nav_status == 'stuck':
            # Physically wedged against something — don't log a bad pair,
            # stop trying for this episode (retrying would likely hit the
            # same obstacle again).
            stuck_in_transit += 1
            log_discard(pair, 'stuck_in_transit')
            break
        if nav_status == 'timeout':
            # Couldn't find a junction this attempt — don't log a bad pair,
            # just stop trying for this episode (autopilot ran out of runway).
            nav_timeouts += 1
            log_discard(pair, 'nav_timeout')
            break

        # ── Close the gap: advance under manual control until actually AT
        # the junction, instead of braking on detection alone ─────────────────
        vehicle.set_autopilot(False)
        approach_status = approach_until_at_junction(world, vehicle, spectator,
                                                       MAX_APPROACH_TICKS)
        total_ticks += MAX_APPROACH_TICKS
        if approach_status == 'stuck':
            approach_stuck += 1
            log_discard(pair, 'approach_stuck')
            vehicle.set_autopilot(True)
            continue
        if approach_status == 'timeout':
            approach_timeouts += 1
            log_discard(pair, 'approach_timeout')
            vehicle.set_autopilot(True)
            continue

        # ── Decision T: YIELD ────────────────────────────────────────────────
        # Sampled per pair, not fixed — see YIELD_SPEED_RANGE comment above.
        yield_speed_target = round(rng.uniform(*YIELD_SPEED_RANGE), 3)
        collision_state['hit'] = False
        hold_position_at_speed(world, vehicle, spectator, TICKS_YIELD,
                                yield_speed_target)
        s_t    = get_state(vehicle, world)
        comp_t = compute_risk_components(vehicle, world)
        rho_t  = comp_t['rho']

        if not s_t['is_junction']:
            # Verification gate: this is the fix for 69% of yield states not
            # being at a real junction in the original run. Discard and let
            # autopilot try again rather than log a bad pair.
            # This is the ONE failure type that can plausibly select on yield
            # speed (a rolling yield can drift past the junction boundary),
            # so its discard row carries the sampled target — comparing that
            # distribution against logged pairs' targets is the direct
            # selection-bias test.
            yield_verify_fails += 1
            log_discard(pair, 'yield_verify_fail', yield_speed_target)
            vehicle.set_autopilot(True)
            continue

        # ── Decision T+1: CROSS ──────────────────────────────────────────────
        cross_junction(world, vehicle, spectator, TICKS_CROSS)
        s_t1    = get_state(vehicle, world)
        comp_t1 = compute_risk_components(vehicle, world)
        rho_t1  = comp_t1['rho']

        off_road = False
        transform = vehicle.get_transform()
        waypoint  = world.get_map().get_waypoint(
            transform.location, project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is not None:
            wp_loc = waypoint.transform.location
            dist   = math.sqrt((transform.location.x - wp_loc.x) ** 2 +
                                (transform.location.y - wp_loc.y) ** 2)
            off_road = dist > 5.0
        else:
            off_road = True

        if collision_state['hit']:
            outcome = 'crash'
        elif off_road:
            outcome = 'off_road'
        else:
            outcome = 'safe'

        # Hand back to autopilot for the next leg
        vehicle.set_autopilot(True)

        writer.writerow({
            'episode_id'    : episode_id,
            't'             : pairs_logged,
            'x_t'           : s_t['x'],
            'y_t'           : s_t['y'],
            'speed_t'       : s_t['speed_ms'],
            'yaw_t'         : s_t['yaw'],
            'lane_id_t'     : s_t['lane_id'],
            'road_id_t'     : s_t['road_id'],
            'is_junction_t' : s_t['is_junction'],
            'action_t'      : 'yield',
            'yield_speed_target_t': yield_speed_target,
            # Component logging (instrumentation pass, 2026-07-12). rho_t was
            # computed at the yield in every previous run but never written —
            # with it logged, T-side risk vs T+1 risk is directly analyzable.
            'junction_progress_t' : comp_t['junction_progress'],
            'speed_risk_t'        : comp_t['speed_risk'],
            'lane_dev_t'          : comp_t['lane_dev'],
            'rho_t'               : rho_t,
            'x_t1'          : s_t1['x'],
            'y_t1'          : s_t1['y'],
            'speed_t1'      : s_t1['speed_ms'],
            'yaw_t1'        : s_t1['yaw'],
            'lane_id_t1'    : s_t1['lane_id'],
            'road_id_t1'    : s_t1['road_id'],
            'is_junction_t1': s_t1['is_junction'],
            'junction_progress_t1': comp_t1['junction_progress'],
            'speed_risk_t1'       : comp_t1['speed_risk'],
            'lane_dev_t1'         : comp_t1['lane_dev'],
            'rho_actual'    : rho_t1,
            'rho_baseline'  : None,
            'RPI_pair'      : None,
            'scenario'      : SCENARIO_TYPE,
            'weather'       : WEATHER,
            'npc_density'   : NPC_DENSITY,
            'outcome'       : outcome,
        })
        pairs_logged += 1
        speed_t_values.append(s_t['speed_ms'])
        rho_actual_values.append(rho_t1)
        component_values.append(comp_t1)

    return (pairs_logged, nav_timeouts, stuck_in_transit, approach_timeouts,
            approach_stuck, yield_verify_fails, speed_t_values, rho_actual_values,
            component_values)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT)
    world  = client.get_world()
    print(f"Connected. Map: {world.get_map().name}")
    print(f"PILOT_MODE = {PILOT_MODE}  (NUM_EPISODES = {NUM_EPISODES})")

    settings                     = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    print("Synchronous mode enabled")

    blueprint_library = world.get_blueprint_library()
    vehicle_bp        = blueprint_library.filter('vehicle.tesla.model3')[0]
    vehicle_bp.set_attribute('color', '0,200,100')
    collision_bp      = blueprint_library.find('sensor.other.collision')

    all_spawns = world.get_map().get_spawn_points()
    spectator  = world.get_spectator()

    fieldnames = [
        'episode_id', 't',
        'x_t', 'y_t', 'speed_t', 'yaw_t', 'lane_id_t', 'road_id_t',
        'is_junction_t', 'action_t', 'yield_speed_target_t',
        'junction_progress_t', 'speed_risk_t', 'lane_dev_t', 'rho_t',
        'x_t1', 'y_t1', 'speed_t1', 'yaw_t1', 'lane_id_t1', 'road_id_t1',
        'is_junction_t1',
        'junction_progress_t1', 'speed_risk_t1', 'lane_dev_t1',
        'rho_actual', 'rho_baseline', 'RPI_pair',
        'scenario', 'weather', 'npc_density', 'outcome',
    ]

    discard_fieldnames = [
        'episode_id', 'pair_attempt', 'failure_type', 'yield_speed_target_t',
        'spawn_x', 'spawn_y', 'fail_x', 'fail_y', 'fail_speed',
    ]

    episode_stats_fieldnames = [
        'episode_id', 'spawn_x', 'spawn_y', 'pairs_logged',
        'nav_timeouts', 'stuck_in_transit', 'approach_timeouts',
        'approach_stuck', 'yield_verify_fails', 'transit_collisions',
        'status',
    ]

    rng = random.Random(RNG_SEED)

    total_pairs              = 0
    total_nav_timeouts       = 0
    total_stuck_in_transit   = 0
    total_approach_timeouts  = 0
    total_approach_stuck     = 0
    total_yield_verify_fails = 0
    total_transit_collisions = 0
    failed_episodes          = []
    all_speed_t              = []   # for the end-of-run saturation sanity check
    all_rho_actual           = []   # for the end-of-run rho_actual variance gate (see get_risk_score fix)
    all_components           = []   # per-pair T+1 component dicts, for the per-component variance gate

    try:
        with open(OUTPUT_FILE, 'w', newline='') as csvfile, \
             open(DISCARD_LOG_FILE, 'w', newline='') as discardfile, \
             open(EPISODE_STATS_FILE, 'w', newline='') as statsfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            discard_writer = csv.DictWriter(discardfile, fieldnames=discard_fieldnames)
            discard_writer.writeheader()
            stats_writer = csv.DictWriter(statsfile, fieldnames=episode_stats_fieldnames)
            stats_writer.writeheader()

            for ep in range(NUM_EPISODES):
                spawn_point      = all_spawns[ep % len(all_spawns)]
                vehicle          = None
                collision_sensor = None
                # 'hit' is reset before each yield-hold (pair-level outcome:
                # did THIS decision pair's window have a collision). Added
                # 2026-07-11: 'total_hits' is NEVER reset within an episode,
                # so it also catches collisions during navigate_to_next_
                # junction() (autopilot transit between junctions) — the
                # previous version only ever checked 'hit', which the
                # pre-yield reset silently wiped if a collision happened
                # during transit. That's how a crash the user could SEE in
                # the CARLA window (car wedged against a lamppost during
                # navigation) produced zero visible signal in the run's
                # outcome/summary numbers.
                collision_state  = {'hit': False, 'total_hits': 0}

                def on_collision(event):
                    collision_state['hit'] = True
                    collision_state['total_hits'] += 1

                try:
                    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                    collision_sensor = world.spawn_actor(
                        collision_bp, carla.Transform(), attach_to=vehicle
                    )
                    collision_sensor.listen(on_collision)

                    print(f"Episode {ep+1}/{NUM_EPISODES} | "
                          f"Spawn: ({spawn_point.location.x:.1f}, "
                          f"{spawn_point.location.y:.1f})",
                          flush=True)

                    # Small settle time
                    for _ in range(10):
                        tick_world(world, vehicle, spectator)

                    (pairs, nav_timeouts, stuck, approach_fails,
                     approach_stuck, yield_fails, ep_speed_t, ep_rho,
                     ep_components) = run_episode(
                        ep + 1, vehicle, world, spectator,
                        writer, discard_writer, collision_state, rng,
                        spawn_point
                    )
                    total_pairs += pairs
                    total_nav_timeouts += nav_timeouts
                    total_stuck_in_transit += stuck
                    total_approach_timeouts += approach_fails
                    total_approach_stuck += approach_stuck
                    total_yield_verify_fails += yield_fails
                    total_transit_collisions += collision_state['total_hits']
                    all_speed_t.extend(ep_speed_t)
                    all_rho_actual.extend(ep_rho)
                    all_components.extend(ep_components)

                    # Per-episode counter persistence (instrumentation pass,
                    # 2026-07-12) — the transit_collisions anomaly could not
                    # be audited because counters only ever existed as console
                    # prints. One row per episode, written immediately.
                    stats_writer.writerow({
                        'episode_id'         : ep + 1,
                        'spawn_x'            : round(spawn_point.location.x, 4),
                        'spawn_y'            : round(spawn_point.location.y, 4),
                        'pairs_logged'       : pairs,
                        'nav_timeouts'       : nav_timeouts,
                        'stuck_in_transit'   : stuck,
                        'approach_timeouts'  : approach_fails,
                        'approach_stuck'     : approach_stuck,
                        'yield_verify_fails' : yield_fails,
                        'transit_collisions' : collision_state['total_hits'],
                        'status'             : 'ok',
                    })
                    print(f"  Episode {ep+1} complete | "
                          f"pairs logged: {pairs} | "
                          f"nav_timeouts: {nav_timeouts} | "
                          f"stuck_in_transit: {stuck} | "
                          f"approach_timeouts: {approach_fails} | "
                          f"approach_stuck: {approach_stuck} | "
                          f"yield_verify_fails: {yield_fails} | "
                          f"transit_collisions: {collision_state['total_hits']} | "
                          f"total so far: {total_pairs}",
                          flush=True)

                except Exception as e:
                    error_msg = f"Episode {ep+1} FAILED: {type(e).__name__}: {e}"
                    print(f"  {error_msg}", flush=True)
                    log_error(error_msg)
                    failed_episodes.append(ep + 1)
                    # Episodes that die mid-run still get a stats row —
                    # counters from the partial episode are lost (run_episode
                    # raised before returning them), but the row itself plus
                    # transit_collisions (tracked outside run_episode) keeps
                    # the per-episode audit trail gap-free.
                    stats_writer.writerow({
                        'episode_id'         : ep + 1,
                        'spawn_x'            : round(spawn_point.location.x, 4),
                        'spawn_y'            : round(spawn_point.location.y, 4),
                        'pairs_logged'       : None,
                        'nav_timeouts'       : None,
                        'stuck_in_transit'   : None,
                        'approach_timeouts'  : None,
                        'approach_stuck'     : None,
                        'yield_verify_fails' : None,
                        'transit_collisions' : collision_state['total_hits'],
                        'status'             : 'exception',
                    })

                finally:
                    if collision_sensor is not None:
                        collision_sensor.stop()
                        collision_sensor.destroy()
                    if vehicle is not None:
                        vehicle.set_autopilot(False)
                        vehicle.destroy()
                    for _ in range(10):
                        world.tick()
                    # Flush + fsync all three outputs after every episode
                    # (2026-07-12). The previous full run's CSV was found
                    # NUL-padded to exactly the byte length of the file it
                    # replaced — a below-the-application write artifact,
                    # root cause still unknown (OneDrive ruled out). This
                    # doesn't explain it, but it makes the output immune to
                    # losing more than one episode's rows to it.
                    for f in (csvfile, discardfile, statsfile):
                        f.flush()
                        os.fsync(f.fileno())

        max_possible = NUM_EPISODES * PAIRS_PER_EPISODE
        print(f"\nAll done. Total pairs: {total_pairs} / {max_possible} possible "
              f"({100*total_pairs/max_possible:.1f}%)")
        print(f"  nav_timeouts (autopilot couldn't find next junction in "
              f"{MAX_NAV_TICKS} ticks): {total_nav_timeouts}")
        print(f"  stuck_in_transit (near-zero speed for "
              f"{STUCK_TICKS_THRESHOLD}+ consecutive ticks under autopilot — "
              f"likely wedged against a static prop): {total_stuck_in_transit}")
        print(f"  approach_timeouts (detected junction, but couldn't reach "
              f"it in {MAX_APPROACH_TICKS} ticks): {total_approach_timeouts}")
        print(f"  approach_stuck (near-zero speed for "
              f"{STUCK_TICKS_THRESHOLD}+ consecutive ticks during the approach "
              f"phase — same wedged-against-a-prop pattern, just relocated "
              f"here after the navigate phase got fixed): {total_approach_stuck}")
        print(f"  yield_verify_fails (reached junction per approach phase, "
              f"but not is_junction after yield hold): {total_yield_verify_fails}")
        print(f"  transit_collisions (collision sensor fired at ANY point in "
              f"the episode, including during autopilot navigation — added "
              f"2026-07-11 because a crash during navigate_to_next_junction() "
              f"was invisible to the old per-pair-only outcome logging): "
              f"{total_transit_collisions}")
        print(f"Data saved to: {OUTPUT_FILE}")
        print(f"Discarded-attempt log: {DISCARD_LOG_FILE}")
        print(f"Per-episode stats: {EPISODE_STATS_FILE}")

        # ── Decision-T variance sanity check (2026-07-11) ───────────────────
        # Gate before trusting this run: the previous version had speed_t
        # identically 0.0000 across all 1461 pairs (zero variance, primary
        # ablation impossible). This run samples YIELD_SPEED_RANGE per pair
        # instead — std should now be meaningfully above 0.
        if all_speed_t:
            mean_speed = statistics.mean(all_speed_t)
            std_speed  = statistics.pstdev(all_speed_t) if len(all_speed_t) > 1 else 0.0
            print(f"\n── Decision-T variance check ({len(all_speed_t)} pairs) ──")
            print(f"  speed_t mean = {mean_speed:.4f} m/s, std = {std_speed:.4f} m/s")
            print(f"  GATE: std should be well above 0 (old full-stop-only run had std=0.0000)")
            if std_speed < 0.3:
                print("  Still looks close to zero-variance — check YIELD_SPEED_RANGE "
                      "and hold_position_at_speed() before running at full scale.")
            else:
                print("  Real variance present — safe to consider flipping PILOT_MODE=False.")
            log_error(f"VARIANCE CHECK: mean={mean_speed:.4f} std={std_speed:.4f}")

        # ── rho_actual variance gate (added 2026-07-12) ─────────────────────
        # This is the actual thing the junction_risk/speed_risk continuity
        # fix was for — the confirmed full-scale defect (2026-07-12) was
        # rho_actual std=0.0141 despite speed_t already having real variance
        # (std=0.74), because junction_risk was pinned at 1.0 and speed_risk
        # was pinned at 0.2 downstream of get_risk_score(). Gate on this
        # BEFORE trusting a rerun's correlation numbers — a std comparable to
        # the old 0.0141 means the fix didn't actually break the pinning.
        if all_rho_actual:
            mean_rho = statistics.mean(all_rho_actual)
            std_rho  = statistics.pstdev(all_rho_actual) if len(all_rho_actual) > 1 else 0.0
            print(f"\n── rho_actual variance check ({len(all_rho_actual)} pairs) ──")
            print(f"  rho_actual mean = {mean_rho:.4f}, std = {std_rho:.4f}")
            print(f"  GATE: std should be well above the old formula's 0.0141 "
                  f"(full-scale, pinned-formula run)")
            if std_rho < 0.03:
                print("  Still looks close to pinned — junction_risk/speed_risk "
                      "continuity fix did not break the plateau. Do not proceed "
                      "to a full rerun; inspect estimate_junction_progress()/"
                      "continuous_speed_risk() output distributions directly "
                      "before changing anything else.")
            else:
                print("  Real variance present in rho_actual — proceed to compute "
                      "the pair/episode-level correlation and permutation test on "
                      "this pilot before considering PILOT_MODE=False.")
            log_error(f"RHO_ACTUAL VARIANCE CHECK: mean={mean_rho:.4f} std={std_rho:.4f}")

        # ── Per-component variance gate (instrumentation pass, 2026-07-12) ──
        # The whole point of this run: rho_actual's variance must be
        # attributable to identifiable components, and each component's
        # spread must be visible on its own. The third audit's offline
        # decomposition found the speed_risk term contributing ~2% of
        # rho_actual's variance (speed_t1 pinned at 5.68±0.09 by controller
        # convergence) — expect that to show up here as a near-zero
        # speed_risk std. That's a KNOWN, accepted pinning (the mechanism
        # question lives in junction_progress vs lane_dev); the gate that
        # matters is that junction_progress_t1 and lane_dev_t1 EACH show
        # real spread, so the decomposition can actually separate them.
        if all_components:
            print(f"\n── Per-component variance check ({len(all_components)} pairs, T+1) ──")
            for key in ('junction_progress', 'speed_risk', 'lane_dev'):
                vals = [c[key] for c in all_components]
                mean_v = statistics.mean(vals)
                std_v  = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                print(f"  {key:18s} mean = {mean_v:.4f}, std = {std_v:.4f}")
                log_error(f"COMPONENT CHECK {key}: mean={mean_v:.4f} std={std_v:.4f}")
            jp_std = statistics.pstdev([c['junction_progress'] for c in all_components])
            ld_std = statistics.pstdev([c['lane_dev'] for c in all_components])
            if jp_std < 0.05:
                print("  GATE FAIL: junction_progress_t1 near-constant — the "
                      "component that carried most of rho_actual's variance in "
                      "the validated run isn't varying here; something is off "
                      "with this run, do not proceed to full scale.")
            elif ld_std < 0.02:
                print("  GATE WARN: lane_dev_t1 near-constant — decomposition "
                      "will attribute everything to junction_progress by "
                      "default rather than by evidence. Usable, but weaker; "
                      "check the raw distribution before deciding.")
            else:
                print("  Both junction_progress and lane_dev show real spread — "
                      "the decomposition can separate them. Proceed to the "
                      "correlation checks on this pilot CSV before flipping "
                      "PILOT_MODE=False.")

        if failed_episodes:
            summary = (f"{len(failed_episodes)}/{NUM_EPISODES} episodes failed: "
                       f"{failed_episodes}")
            print(summary, flush=True)
            log_error(f"RUN SUMMARY: {summary}")
        else:
            log_error(f"RUN SUMMARY: all {NUM_EPISODES} episodes completed cleanly.")

    finally:
        print("Restoring asynchronous mode...")
        settings                  = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Done.")


if __name__ == '__main__':
    main()
