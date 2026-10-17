"""
Scenario A variant: highway lane change WITH NPC TRAFFIC + RSS lateral-margin
logging.

Purpose (see CLAUDE.md "RSS baseline" and "NPC-traffic scope" decision
entries, 2026-07-14): the same two standing reviewer objections answered for
Scenario C, now for the LATERAL case RSS was actually built for —
  1. "Empty world": adds Traffic Manager vehicles as ambient traffic, plus a
     fresh escort NPC placed in the TARGET lane for each individual decision
     pair (not once per episode — see design note below on why A needs this
     at finer granularity than C did).
  2. "Hand-crafted risk formulas": alongside the UNCHANGED Scenario A rho,
     this run logs an RSS-based LATERAL safe-distance margin to the nearest
     vehicle in the lane the ego is changing INTO, at both T and T+1. RSS is
     memoryless by construction; if decision-T steer severity predicts the
     lateral margin at T+1 (same episode-level ablation machinery as
     analyse_scenario_c_npc_rss.py), carry-forward is demonstrated on this
     scenario's most natural externally-defined comparator — RSS's lateral/
     cut-in case is literally what a lane change is.

LATERAL RSS FORMULA — verified against the primary source before writing any
of this, not reconstructed from memory (the longitudinal case used in
scenario_c_npc_rss.py is well-known enough to recall reliably; the lateral
case is not, and this is going in a paper):
  Lemma 4, Shalev-Shwartz, Shashua, Shammah, "On a Formal Model of Safe and
  Scalable Self-driving Cars," arXiv:1708.06374 (2017), Section 3.4.
  With c1 to the left of c2, lateral velocities v1, v2 (same global lateral
  axis, positive = rightward, NOT "toward the other car" — that sign
  convention falls out of the left/right assignment itself):
    v1p = v1 + rho * a_lat_max_accel
    v2p = v2 - rho * a_lat_max_accel
    d_min = mu + max(0,
              (v1+v1p)/2 * rho + v1p^2/(2*a_lat_min_brake)
            - (v2+v2p)/2 * rho - v2p^2/(2*a_lat_min_brake))
  IMPORTANT SCOPING DISCLAIMER for the paper: this implements the STANDALONE
  lateral safe-distance case (Definition 6 / Lemma 4) applied to the nearest
  longitudinally-proximate vehicle in the target lane. It is NOT the full
  combined longitudinal+lateral proper-response formalism of the source
  paper's Section 3.5. Label this "RSS-based lateral safe-distance margin"
  in the paper — never "the full RSS safety envelope" or "the CARLA RSS
  sensor" (which is Linux-only and unused here, same constraint as C's
  variant).

DESIGN NOTE — why A's escort NPCs are spawned per PAIR, not per episode
(unlike C): Scenario C's decision is a single fixed direction (braking) for
the whole episode, so one escort placed near the episode's spawn point stays
relevant across all 8 pairs. Scenario A alternates lane-change direction
every pair (left, right, left, ...) and the TARGET lane is resolved
dynamically per pair via resolve_target_waypoint() (unchanged, banked
logic) — a single episode-start escort would only ever be in the right lane
for at most half the pairs, by construction. Escorts here are spawned in the
CURRENTLY-RESOLVED target lane immediately before each pair's steer phase,
and destroyed right after that pair's T+1 sample, before the next pair
(whose target lane may be the opposite side).

Approach B preserved exactly as in the banked script and in
scenario_c_npc_rss.py: steer_mag/steer_ticks severity is sampled and applied
on the same fixed per-pair schedule regardless of traffic; NPCs (ambient
pool + per-pair escort) never gate or alter the decision itself. The only
traffic-aware control is a longitudinal headway guard during cruise/warmup/
relocate phases (transit, not a decision) — same justification as C's
headway guard, reusing the same geometric+heading-filtered neighbor
detection pattern (with the heading-misclassification fix already applied
from day one here, not retrofitted after a buggy first pilot like C's was).

rho_actual math (compute_risk_components, get_state, all steering/cruise
control) is BYTE-IDENTICAL to the banked scenario_a_lanechange.py — RSS
columns are ADDITIONS. Never merge this file's output with the banked one.

Output: scenario_a_lanechange_npc_rss[_pilot].csv + discard log + episode
stats sidecar. The banked scenario_a_lanechange.csv cannot be overwritten by
this script.

Run:  py -3.7 scripts\\scenario_a_lanechange_npc_rss.py
"""
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

PILOT_MODE        = False
NUM_EPISODES      = 20 if PILOT_MODE else 100
# ──────────────────────────────────────────────────────────────────────────────

PAIRS_PER_EPISODE = 8
WARMUP_TICKS      = 80
TICKS_SETTLE      = 40

# Severity sampling — IDENTICAL ranges to the banked script, metric locked
# a priori (steer_mag * steer_ticks). Do not recalibrate these here; they
# were already pilot-validated on the no-NPC run.
STEER_MAG_RANGE   = (0.05, 0.20)
STEER_TICKS_RANGE = (10, 25)

TICKS_STABILISE_SAMPLE = 15    # T+1 mid-transient — unchanged from banked

CRUISE_SPEED      = 8.0
CRUISE_LOOKAHEAD  = 6.0
OPTIMAL_SPEED     = 7.0
SPEED_RISK_SCALE  = 8.0
LAT_DEV_SCALE     = 1.5
HEADING_ERR_SCALE = 20.0

MAX_RELOCATE_TRIES = 3
RELOCATE_TICKS     = 50

# ── Steering-validity limits (added 2026-07-22, fourth pilot) ───────────────
# _steer_toward() returns yaw_diff/pi*2 clamped to +-1, so a target waypoint
# BEHIND the ego (yaw_diff ~ 180 deg) commands FULL STEERING LOCK — a U-turn.
# Two things produce a behind-target: (a) waypoint.next()[0] blindly taking the
# first successor, which at a junction can be a sharp/reversing branch, and
# (b) an overshooting lane change putting the ego physically in the OPPOSING
# lane, after which project_to_road snaps it to that lane and next() points
# backwards. Case (b) is what produces "random U-turn into oncoming traffic".
# This latent defect exists in the banked no-NPC script too; an empty world
# simply had nothing to collide with, so it only ever showed up as a large
# lat_dev rather than a crash.
WAYPOINT_DIR_MAX_DIFF = 90.0   # deg — a candidate/projected lane whose
                               # direction differs from the ego's heading by
                               # more than this is not a valid continuation.
                               # 90 deg is the principled boundary, not a tuned
                               # value: beyond it the "forward" direction of
                               # that lane has a negative component along the
                               # ego's heading, i.e. following it means
                               # reversing.

# ── Clean-start preconditions for a decision pair ───────────────────────────
# The decision itself stays traffic-blind (Approach B). What is gated here is
# WHEN a pair is allowed to BEGIN — a transit-phase concern, exactly like
# Scenario B using autopilot to reach a junction before scripting the yield,
# and Scenario C's cruise-phase headway guard. Without this the ego can enter
# a scripted merge already boxed in (it never stops at traffic lights, so it
# drives into queued traffic), and the resulting collision measures the
# starting position, not the lane-change decision.
# NARROWED 2026-07-22 after the escort-disabled pilot. The first version used
# radius 25 m and a 6 m lateral band. A lane is ~3.5 m wide, so a 6 m band
# spans the ego's lane AND both neighbours, while a decision only covers
# ~12-16 m of travel — meaning the gate structurally guaranteed the TARGET
# LANE was empty for the whole decision. Measured consequence: RSS coverage
# fell to 1.9% of pairs (2/108, in 2/18 episodes) against 54-58% when escorts
# were supplying the target-lane vehicle. The gate was making the very
# phenomenon this variant exists to observe impossible to observe, and it is
# also what produced the "ego just tails an NPC and never acts" behaviour
# (29 no_clear_start timeouts in that pilot).
# The gate's LEGITIMATE purpose is narrower: do not begin a scripted merge
# while already in a car-following conflict in the ego's OWN lane, because
# the resulting collision would measure the starting position rather than the
# lane-change decision. Traffic in the TARGET lane is the phenomenon under
# study, not a hazard to be excluded.
CLEAR_START_RADIUS   = 12.0   # m — matches HEADWAY_BRAKE_GAP (defined below),
                              # i.e. exactly the distance at which the ego's
                              # own headway guard starts braking. Reusing that
                              # existing, already-justified threshold rather
                              # than inventing a second unrelated constant:
                              # inside it the ego is by definition already in
                              # a following conflict.
CLEAR_START_LATERAL  = 2.0    # m — the ego's OWN lane only. Same tolerance
                              # nearest_ahead_same_lane() uses for "same lane".
CLEAR_START_MAX_WAIT = 200    # ticks (~10 s) before giving up on this pair

RNG_SEED           = 42     # decision severity sampling ONLY
NPC_RNG_SEED        = 43    # ambient NPC pool selection — separate from RNG_SEED
                            # (2026-07-21 lesson from scenario_c_npc_rss.py:
                            # never let spawn-mechanics randomness perturb the
                            # severity draw sequence)
ESCORT_RNG_SEED     = 44    # per-pair escort selection — also separate

# ── NPC traffic: ambient pool (background, general realism) ──────────────────
N_NPC              = 30
TM_PORT            = 8000
TM_SEED            = 42
NPC_SPAWN_RADIUS   = 100.0   # m — bias ambient pool toward ego's operating area
                             # (see scenario_c_npc_rss.py REVISION note — a
                             # pool spawned once for the whole run and left to
                             # wander does NOT reliably stay near an ego that
                             # relocates constantly; this bias alone is not
                             # expected to carry RSS coverage, the per-pair
                             # escort below does that job)

# ── Escort NPC: per-pair, placed in the resolved TARGET lane ─────────────────
# REWRITTEN 2026-07-22 after the second pilot showed a 16/20-episode crash rate
# and the user observed NPCs "suddenly spawning in front of" the ego. Confirmed
# from the pilot CSV, not assumed: crash pairs had mean lateral gap 0.83 m at T
# (min 0.00) vs 2.22 m on safe pairs, and 14/16 crashes had an RSS neighbor
# present. Root cause was the old map-spawn-point approach — candidates were
# accepted anywhere in |longi| <= 20 m, including 1-2 m directly ahead of the
# ego in the very lane it was about to steer into. That manufactures collisions
# rather than measuring them.
#
# The escort is now placed by walking the TARGET LANE's own centreline a
# controlled distance from the ego (waypoint.next()/previous()), not by hunting
# for a nearby map spawn point. This fixes three things at once: (1) exact
# control over separation, (2) guaranteed correct lane alignment and heading,
# (3) far better availability — no longer dependent on sparse spawn-point
# placement, which was also capping escort coverage at ~40% of pairs.
# ── ESCORT DISABLED 2026-07-22, after the 100-episode run ───────────────────
# The per-pair escort existed for ONE reason: to guarantee a vehicle in the
# target lane so an RSS lateral margin could be computed. That RSS layer was
# dropped (see CLAUDE.md 2026-07-22 — both margins are design-level
# confounded), which leaves the escort with no purpose while it actively
# destroys the run's actual objective.
#
# Evidence from the 100-episode run: 59 of 60 crashes had an escort placed;
# escort-AHEAD crashed 36% of the time vs 1% for escort-BEHIND (TM drives the
# escort at the posted limit while the ego holds a fixed CRUISE_SPEED, so on
# slower roads the ego simply consumes the 10-16 m gap). Those crashes end
# episodes early: pair yield fell to 44% (355/800), mean 3.86 pairs/episode
# against the designed 8, with 34/92 episodes down to <= 2 pairs. Episode
# means over one or two pairs are almost pure noise, which is why the primary
# ablation came back r = -0.234 with a bootstrap CI of [-0.539, +0.094] —
# crossing zero, i.e. failing this project's standing significance bar,
# despite the banked no-NPC result being a clean -0.495.
#
# The remaining purpose of this variant — "does A's carry-forward hold with
# traffic present?" — is served by the 22-vehicle ambient TM pool, which is
# real traffic the ego must contend with and is NOT geometry we manufactured.
# Disabling the escort removes the dominant crash source, restores pair yield
# toward the banked run's 82%, and drops the "you staged the conflict"
# criticism entirely. Set back to True only if the RSS layer is ever revived.
ESCORT_ENABLED = False

ESCORT_SPAWN_DIST_RANGE = (10.0, 16.0)  # m — longitudinal offset from the ego,
                              # sampled per pair. Deliberate trade-off: a
                              # perfectly abeam escort maximises lateral-RSS
                              # relevance but guarantees a collision when a
                              # blind scripted merge fires, killing the episode
                              # and the remaining pairs. 10-16 m is a realistic
                              # leading/following separation that keeps the
                              # escort inside NEIGHBOR_LONGI_WINDOW (so the RSS
                              # lateral margin is measurable) without staging
                              # a crash. Crashes that still occur are then
                              # legitimate outcomes of a weak/late merge, which
                              # is exactly what the scenario is meant to test.
ESCORT_AHEAD_PROB    = 0.5    # split ahead/behind — both are real merge
                              # conflicts and RSS's lateral case is symmetric
ESCORT_CLEARANCE     = 6.0    # m — refuse to spawn if any vehicle is already
                              # within this radius of the intended point
                              # (prevents stacking an escort onto an ambient
                              # NPC, which try_spawn_actor alone won't catch
                              # until the bounding boxes actually overlap)
ESCORT_SPEED_PCT     = 0.0    # TM speed offset (%) — 0 = follow the posted
                              # limit, roughly matching the ego's CRUISE_SPEED
                              # so relative longitudinal drift stays small over
                              # a pair's ~1-2 s lifetime

# ── RSS lateral parameters (Lemma 4, Shalev-Shwartz et al. 2017 — see module
# docstring for the exact equation and citation) ─────────────────────────────
RSS_LAT_RHO           = 1.0   # s — response time (kept consistent with the
                              # longitudinal case in scenario_c_npc_rss.py)
RSS_LAT_A_MAX_ACCEL   = 2.0   # m/s^2 — max lateral acceleration during response
                              # (lower than the longitudinal 3.5 — lateral
                              # acceleration capability is physically smaller)
RSS_LAT_B_MIN_BRAKE   = 3.0   # m/s^2 — min guaranteed lateral braking
RSS_LAT_MU            = 0.2   # m — minimal lateral buffer (Definition 5/6's mu)
# NOTE: the source paper defines the FORMULA structure, not universal numeric
# values for these physical parameters — actual magnitudes depend on vehicle/
# regulatory assumptions chosen by the implementer, same epistemic status as
# RSS_A_MAX_ACCEL etc. already used for the longitudinal case in
# scenario_c_npc_rss.py. Disclose as "assumed physical parameters" in the paper.

# ── RSS LONGITUDINAL parameters (Lemma 2 / Definition 1, same source paper) ──
# Values deliberately identical to scenario_c_npc_rss.py so the longitudinal
# margins produced by the two NPC variants are directly comparable — a
# different parameter set would make "does A agree with C" unanswerable.
# Added 2026-07-22 (fourth pilot): the lateral margin alone is structurally
# confounded here. Pilot data — completed merges gap 0.95 m / margin -1.49,
# failed merges gap 1.75 m / margin -2.14 — shows severity pushes the lateral
# margin in OPPOSITE directions via two channels (completing shrinks the gap;
# failing leaves a live lateral transient that inflates d_min), which cancel to
# r=+0.067. It also yields a 74% "violation" rate, because two cars in
# adjacent lanes are almost always laterally unsafe by the standalone lateral
# criterion. That is precisely why RSS never uses it alone: the source paper's
# combined proper response (Sec 3.5) calls a situation dangerous only when it
# is BOTH longitudinally and laterally dangerous. Logging both margins lets the
# analysis use the case that is actually binding — after a completed merge the
# relevant question is the following distance the merge left behind — and lets
# it compute the real combined RSS criterion.
RSS_LONG_RHO          = 1.0   # s      — response time (matches C's variant)
RSS_LONG_A_MAX_ACCEL  = 3.5   # m/s^2  — max accel during response
RSS_LONG_B_MIN_BRAKE  = 4.0   # m/s^2  — min guaranteed braking, rear vehicle
RSS_LONG_B_MAX_BRAKE  = 8.0   # m/s^2  — max braking, front vehicle
VEHICLE_LENGTH_COMP   = 4.5   # m      — bumper-to-bumper correction

NEIGHBOR_LONGI_WINDOW = 20.0  # m — how far ahead/behind (along travel
                              # direction) a target-lane vehicle can be and
                              # still count as relevant to THIS merge decision.
                              # Widened 15 -> 20 (2026-07-22) to sit above
                              # ESCORT_SPAWN_DIST_RANGE's upper bound plus a
                              # margin for relative drift during a pair — a
                              # detection window narrower than the spawn range
                              # would silently drop the escort from the RSS
                              # snapshot precisely when it was placed furthest.
VEHICLE_WIDTH_COMP    = 2.0   # m — combined half-widths, lateral analog of
                              # scenario_c_npc_rss.py's VEHICLE_LENGTH_COMP
HEADING_MAX_DIFF      = 30.0  # deg — reject oncoming/cross traffic from
                              # being read as a target-lane neighbor (same
                              # fix class as scenario_c_npc_rss.py, applied
                              # from the start here instead of retrofitted)

# Headway guard (cruise/warmup/relocate phases ONLY — transit, not a decision;
# same justification as scenario_c_npc_rss.py's headway guard):
HEADWAY_BRAKE_GAP  = 12.0
HEADWAY_FULL_GAP   = 5.0

OUTPUT_DIR         = r'C:\Users\jayad\Desktop\RPI_AINN\data'
OUTPUT_FILE        = os.path.join(
    OUTPUT_DIR,
    'scenario_a_lanechange_npc_rss_pilot.csv' if PILOT_MODE
    else 'scenario_a_lanechange_npc_rss.csv'
)
SCENARIO_TYPE       = 'A_lanechange_npc_rss'
WEATHER             = 'clear'
ERROR_LOG_FILE      = OUTPUT_FILE.replace('.csv', '_errors.log')
DISCARD_LOG_FILE    = OUTPUT_FILE.replace('.csv', '_discards.csv')
EPISODE_STATS_FILE  = OUTPUT_FILE.replace('.csv', '_episode_stats.csv')
# ──────────────────────────────────────────────────────────────────────────────


def log_error(msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")


def pearson_r(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def soft_cap(x, scale):
    return 1.0 - math.exp(-x / scale)


# ── Target-lane machinery — UNCHANGED from banked scenario_a_lanechange.py ──

def adjacent_lane(waypoint, direction):
    nb = waypoint.get_left_lane() if direction == 'left' else waypoint.get_right_lane()
    if nb is None:
        return None
    if nb.lane_type != carla.LaneType.Driving:
        return None
    if nb.lane_id * waypoint.lane_id < 0:
        return None
    return nb


def lane_direction_ok(wp, ref_yaw):
    """True if `wp`'s lane runs broadly the same way the ego is facing. Guards
    against project_to_road snapping an overshooting ego onto the OPPOSING
    lane, whose next() chain points backwards (see WAYPOINT_DIR_MAX_DIFF)."""
    return _heading_diff_deg(wp.transform.rotation.yaw, ref_yaw) <= WAYPOINT_DIR_MAX_DIFF


def best_next_waypoint(wp, dist, ref_yaw):
    """`wp.next(dist)` can return several successors (junctions especially);
    the old code always took [0], which is an arbitrary branch and may point
    sharply away from — or back along — the ego's heading. Pick the candidate
    whose direction best matches ref_yaw, and reject the lot if even the best
    one would require reversing."""
    cands = wp.next(dist)
    if not cands:
        return None
    best = min(cands, key=lambda c: _heading_diff_deg(c.transform.rotation.yaw, ref_yaw))
    if not lane_direction_ok(best, ref_yaw):
        return None
    return best


def resolve_target_waypoint(world, vehicle, original_lane_id, direction):
    ego_yaw = vehicle.get_transform().rotation.yaw
    wp = world.get_map().get_waypoint(
        vehicle.get_transform().location,
        project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if wp is None:
        return None, False
    if not lane_direction_ok(wp, ego_yaw):
        # Ego has ended up on an opposing lane — there is no valid target to
        # steer toward. Returning None makes compute_risk_components fall into
        # its "no resolvable target" branch (high risk, correctly) and stops
        # target_lane_steer from commanding a U-turn to follow it.
        return None, False
    if wp.lane_id == original_lane_id:
        return adjacent_lane(wp, direction), False
    return wp, True


def is_wrong_way(world, vehicle):
    """Ego physically on a lane running against its heading. Reported as an
    outcome instead of being silently steered out of, because the controller's
    only 'fix' is a full-lock U-turn into oncoming traffic."""
    wp = world.get_map().get_waypoint(
        vehicle.get_transform().location,
        project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if wp is None:
        return False
    return not lane_direction_ok(wp, vehicle.get_transform().rotation.yaw)


def compute_risk_components(vehicle, world, original_lane_id, direction):
    """UNCHANGED from banked scenario_a_lanechange.py — byte-identical math."""
    transform = vehicle.get_transform()
    velocity  = vehicle.get_velocity()
    speed_ms  = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    target_wp, _ = resolve_target_waypoint(world, vehicle, original_lane_id, direction)

    if target_wp is None:
        lat_dev_m   = LAT_DEV_SCALE * 3.0
        heading_deg = HEADING_ERR_SCALE * 3.0
    else:
        t_loc     = target_wp.transform.location
        lat_dev_m = math.sqrt((transform.location.x - t_loc.x) ** 2 +
                              (transform.location.y - t_loc.y) ** 2)
        diff = (transform.rotation.yaw - target_wp.transform.rotation.yaw) % 360.0
        heading_deg = min(diff, 360.0 - diff)

    lat_dev     = soft_cap(lat_dev_m, LAT_DEV_SCALE)
    heading_err = soft_cap(heading_deg, HEADING_ERR_SCALE)
    deviation   = (speed_ms - OPTIMAL_SPEED) / SPEED_RISK_SCALE
    speed_risk  = min(1.0, 0.10 + deviation ** 2)

    return {
        'lat_dev'    : round(lat_dev, 6),
        'heading_err': round(heading_err, 6),
        'speed_risk' : round(speed_risk, 6),
        'rho'        : round(0.5 * lat_dev + 0.3 * heading_err + 0.2 * speed_risk, 6),
        'lat_dev_m'  : round(lat_dev_m, 4),
        'heading_deg': round(heading_deg, 4),
    }


def get_state(vehicle, world):
    transform = vehicle.get_transform()
    velocity  = vehicle.get_velocity()
    speed_ms  = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    waypoint  = world.get_map().get_waypoint(
        transform.location, project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    return {
        'x': round(transform.location.x, 4), 'y': round(transform.location.y, 4),
        'yaw': round(transform.rotation.yaw, 4), 'speed_ms': round(speed_ms, 4),
        'lane_id': waypoint.lane_id if waypoint else None,
        'road_id': waypoint.road_id if waypoint else None,
    }


def lane_follow_steer(vehicle, world, lookahead):
    transform = vehicle.get_transform()
    ego_yaw   = transform.rotation.yaw
    waypoint  = world.get_map().get_waypoint(
        transform.location, project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        return 0.0
    if not lane_direction_ok(waypoint, ego_yaw):
        return 0.0   # opposing-lane projection: hold straight rather than
                     # commanding a full-lock turn to chase a backwards lane
    ahead = best_next_waypoint(waypoint, lookahead, ego_yaw)
    if ahead is None:
        return 0.0
    return _steer_toward(vehicle, ahead.transform.location)


def target_lane_steer(vehicle, world, original_lane_id, direction, lookahead):
    target_wp, _ = resolve_target_waypoint(world, vehicle, original_lane_id, direction)
    if target_wp is None:
        return lane_follow_steer(vehicle, world, lookahead)
    ego_yaw = vehicle.get_transform().rotation.yaw
    ahead = best_next_waypoint(target_wp, lookahead, ego_yaw)
    if ahead is None:
        return 0.0
    return _steer_toward(vehicle, ahead.transform.location)


def _steer_toward(vehicle, target_loc):
    transform   = vehicle.get_transform()
    vehicle_yaw = math.radians(transform.rotation.yaw)
    dx = target_loc.x - transform.location.x
    dy = target_loc.y - transform.location.y
    yaw_diff = math.atan2(dy, dx) - vehicle_yaw
    while yaw_diff >  math.pi: yaw_diff -= 2 * math.pi
    while yaw_diff < -math.pi: yaw_diff += 2 * math.pi
    return max(-1.0, min(1.0, yaw_diff / math.pi * 2.0))


def _speed_of(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def _heading_diff_deg(yaw_a, yaw_b):
    diff = abs(yaw_a - yaw_b) % 360.0
    return min(diff, 360.0 - diff)


# ── RSS lateral machinery ────────────────────────────────────────────────────

heading_stats = {'checked': 0, 'rejected': 0}   # diagnostic, printed at run end
# Module-level so every episode contributes to one run-total; see Gate 3b.
rss_present_count = [0]


def rss_d_min_longitudinal(v_rear, v_front):
    """RSS minimum safe LONGITUDINAL distance (m), same formula and parameters
    as scenario_c_npc_rss.py::rss_d_min so the two variants are comparable."""
    d = (v_rear * RSS_LONG_RHO
         + 0.5 * RSS_LONG_A_MAX_ACCEL * RSS_LONG_RHO ** 2
         + (v_rear + RSS_LONG_RHO * RSS_LONG_A_MAX_ACCEL) ** 2 / (2.0 * RSS_LONG_B_MIN_BRAKE)
         - v_front ** 2 / (2.0 * RSS_LONG_B_MAX_BRAKE))
    return max(0.0, d)


def rss_d_min_lateral(v1, v2, rho, a_max_accel, a_min_brake, mu):
    """Lemma 4, Shalev-Shwartz et al. 2017 (see module docstring for the
    equation and exact citation). c1 assumed to the left of c2; v1, v2 are
    both on the SAME global lateral axis (positive = rightward), not
    "toward the other car" — the left/right assignment is what encodes
    direction of approach."""
    v1p = v1 + rho * a_max_accel
    v2p = v2 - rho * a_max_accel
    raw = ((v1 + v1p) / 2.0 * rho + v1p ** 2 / (2.0 * a_min_brake)
           - (v2 + v2p) / 2.0 * rho - v2p ** 2 / (2.0 * a_min_brake))
    return mu + max(0.0, raw)


def nearest_target_lane_neighbor(world, vehicle, original_lane_id, direction):
    """
    Nearest vehicle currently on the TARGET lane (the lane this pair's
    maneuver is headed into), within a longitudinal window, same direction
    of travel (heading-filtered — see scenario_c_npc_rss.py for why this
    matters: without it, an oncoming vehicle in an unrelated lane can be
    geometrically close enough to get picked up).
    Returns None or a dict with gap/velocities needed for the RSS lateral
    calc.
    """
    target_wp, _ = resolve_target_waypoint(world, vehicle, original_lane_id, direction)
    if target_wp is None:
        return None

    ego_tf   = vehicle.get_transform()
    ego_loc  = ego_tf.location
    ego_yaw  = ego_tf.rotation.yaw
    fwd      = ego_tf.get_forward_vector()
    right    = ego_tf.get_right_vector()
    ego_v    = vehicle.get_velocity()
    # ego velocity resolved onto its own current right / forward axes
    ego_lat_v  = ego_v.x * right.x + ego_v.y * right.y
    ego_long_v = ego_v.x * fwd.x + ego_v.y * fwd.y

    best = None
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.id == vehicle.id:
            continue
        actor_tf = actor.get_transform()
        wp = world.get_map().get_waypoint(
            actor_tf.location, project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if wp is None or wp.road_id != target_wp.road_id or wp.lane_id != target_wp.lane_id:
            continue   # not actually on the target lane right now
        heading_stats['checked'] += 1
        if _heading_diff_deg(actor_tf.rotation.yaw, ego_yaw) > HEADING_MAX_DIFF:
            heading_stats['rejected'] += 1
            continue
        rel_x = actor_tf.location.x - ego_loc.x
        rel_y = actor_tf.location.y - ego_loc.y
        longi = rel_x * fwd.x + rel_y * fwd.y
        if abs(longi) > NEIGHBOR_LONGI_WINDOW:
            continue
        lat_offset = rel_x * right.x + rel_y * right.y   # signed: + = to ego's right
        gap = max(0.0, abs(lat_offset) - VEHICLE_WIDTH_COMP)
        actor_v = actor.get_velocity()
        actor_lat_v  = actor_v.x * right.x + actor_v.y * right.y
        actor_long_v = actor_v.x * fwd.x + actor_v.y * fwd.y
        if best is None or gap < best['gap']:
            best = {'gap': gap, 'lat_offset': lat_offset,
                    'ego_lat_v': ego_lat_v, 'npc_lat_v': actor_lat_v,
                    'longi': longi,
                    'long_gap': max(0.0, abs(longi) - VEHICLE_LENGTH_COMP),
                    'ego_long_v': ego_long_v, 'npc_long_v': actor_long_v}
    return best


def rss_lateral_snapshot(world, vehicle, original_lane_id, direction):
    """Raw values logged, no baked-in normalization — analysis decides
    offline, per house rules on capped/derived components."""
    nb = nearest_target_lane_neighbor(world, vehicle, original_lane_id, direction)
    snap = {'present': nb is not None, 'gap': None,
            'ego_lat_v': None, 'npc_lat_v': None, 'margin': None,
            'longi': None, 'long_gap': None, 'long_margin': None,
            'rss_dangerous': None}
    if nb is not None:
        # ── Lateral case (Lemma 4) ──────────────────────────────────────
        # Assign c1 (left) / c2 (right) by sign of lat_offset (ego's right-
        # axis convention: positive lat_offset means the neighbor is to the
        # ego's right, i.e. ego is c1/left, neighbor is c2/right).
        if nb['lat_offset'] >= 0:
            v1, v2 = nb['ego_lat_v'], nb['npc_lat_v']
        else:
            v1, v2 = nb['npc_lat_v'], nb['ego_lat_v']
        dmin_lat = rss_d_min_lateral(v1, v2, RSS_LAT_RHO, RSS_LAT_A_MAX_ACCEL,
                                     RSS_LAT_B_MIN_BRAKE, RSS_LAT_MU)
        lat_margin = nb['gap'] - dmin_lat

        # ── Longitudinal case (rear vehicle is whichever is behind) ─────
        if nb['longi'] >= 0:      # neighbor ahead -> ego is the rear vehicle
            dmin_long = rss_d_min_longitudinal(nb['ego_long_v'], nb['npc_long_v'])
        else:                     # neighbor behind -> neighbor is the rear
            dmin_long = rss_d_min_longitudinal(nb['npc_long_v'], nb['ego_long_v'])
        long_margin = nb['long_gap'] - dmin_long

        snap.update(gap=round(nb['gap'], 3),
                    ego_lat_v=round(nb['ego_lat_v'], 3),
                    npc_lat_v=round(nb['npc_lat_v'], 3),
                    margin=round(lat_margin, 3),
                    longi=round(nb['longi'], 3),
                    long_gap=round(nb['long_gap'], 3),
                    long_margin=round(long_margin, 3),
                    # The ACTUAL RSS criterion (source paper Sec 3.5): a
                    # situation is dangerous only when BOTH axes are unsafe.
                    # Logged as a flag so the analysis never has to rely on
                    # the standalone lateral violation rate, which is ~74%
                    # here purely because adjacent-lane driving is always
                    # laterally "unsafe" in isolation.
                    rss_dangerous=(lat_margin < 0 and long_margin < 0))
    return snap


# ── Longitudinal neighbor detection — headway guard only (transit phases) ───

def nearest_ahead_same_lane(world, vehicle):
    """Simplified, front-only, heading-filtered same-lane-ahead detector,
    used ONLY by the headway guard during cruise/warmup/relocate (transit,
    not a decision — same justification as scenario_c_npc_rss.py)."""
    ego_tf  = vehicle.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = ego_tf.rotation.yaw
    fwd     = ego_tf.get_forward_vector()
    right   = ego_tf.get_right_vector()
    best = None
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.id == vehicle.id:
            continue
        actor_tf = actor.get_transform()
        loc = actor_tf.location
        rel_x, rel_y = loc.x - ego_loc.x, loc.y - ego_loc.y
        longi = rel_x * fwd.x + rel_y * fwd.y
        lat   = abs(rel_x * right.x + rel_y * right.y)
        if longi <= 0.5 or longi > 40.0 or lat > 2.0:
            continue
        if _heading_diff_deg(actor_tf.rotation.yaw, ego_yaw) > HEADING_MAX_DIFF:
            continue
        gap = max(0.0, longi - 4.5)
        if best is None or gap < best:
            best = gap
    return best


# ── Control loops (banked patterns + headway guard) ──────────────────────────

def tick_world(world, vehicle, spectator):
    world.tick()
    loc   = vehicle.get_transform().location
    yaw   = vehicle.get_transform().rotation.yaw
    fwd_x = math.cos(math.radians(yaw))
    fwd_y = math.sin(math.radians(yaw))
    spectator.set_transform(carla.Transform(
        carla.Location(x=loc.x - 12 * fwd_x, y=loc.y - 12 * fwd_y, z=loc.z + 7),
        carla.Rotation(pitch=-25, yaw=yaw)
    ))


def speed_throttle(vehicle, target_speed):
    velocity = vehicle.get_velocity()
    speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return max(0.0, min(0.75, 0.35 + (target_speed - speed_ms) * 0.10))


def cruise(world, vehicle, spectator, n_ticks):
    """Steady-state lane keeping WITH a longitudinal headway guard (transit
    only — see module docstring)."""
    for _ in range(n_ticks):
        steer = lane_follow_steer(vehicle, world, CRUISE_LOOKAHEAD)
        gap = nearest_ahead_same_lane(world, vehicle)
        ctrl = carla.VehicleControl()
        ctrl.steer = steer
        if gap is not None and gap < HEADWAY_BRAKE_GAP:
            if gap <= HEADWAY_FULL_GAP:
                ctrl.throttle, ctrl.brake = 0.0, 1.0
            else:
                frac = (HEADWAY_BRAKE_GAP - gap) / (HEADWAY_BRAKE_GAP - HEADWAY_FULL_GAP)
                ctrl.throttle, ctrl.brake = 0.0, min(1.0, frac)
        else:
            ctrl.throttle, ctrl.brake = speed_throttle(vehicle, CRUISE_SPEED), 0.0
        vehicle.apply_control(ctrl)
        tick_world(world, vehicle, spectator)


def start_is_clear(world, vehicle):
    """Preconditions for BEGINNING a pair: not inside a junction, and no
    pre-existing vehicle close ahead. This gates when the decision starts, not
    how it executes — the scripted steer remains entirely traffic-blind. Same
    principle already accepted for Scenario B (autopilot navigates to the
    junction; only the yield/cross are scripted) and Scenario C (cruise-phase
    headway guard). The escort has not been spawned yet at call time, so this
    never rejects a pair because of the vehicle we are about to place."""
    tf  = vehicle.get_transform()
    wp  = world.get_map().get_waypoint(tf.location, project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
    if wp is None or wp.is_junction:
        return False
    fwd, right = tf.get_forward_vector(), tf.get_right_vector()
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.id == vehicle.id:
            continue
        loc = actor.get_transform().location
        rel_x, rel_y = loc.x - tf.location.x, loc.y - tf.location.y
        longi = rel_x * fwd.x + rel_y * fwd.y
        lat   = abs(rel_x * right.x + rel_y * right.y)
        if 0.0 < longi < CLEAR_START_RADIUS and lat < CLEAR_START_LATERAL:
            return False
    return True


def wait_for_clear_start(world, vehicle, spectator):
    """Cruise until start_is_clear(), or give up. Returns True if clear."""
    waited = 0
    while waited < CLEAR_START_MAX_WAIT:
        if start_is_clear(world, vehicle):
            return True
        cruise(world, vehicle, spectator, 10)
        waited += 10
    return False


def execute_steer_phase(world, vehicle, spectator, steer_mag, steer_ticks, direction):
    """Decision T — UNCHANGED, traffic-blind on purpose."""
    steer_sign = -1.0 if direction == 'left' else 1.0
    for _ in range(steer_ticks):
        ctrl          = carla.VehicleControl()
        ctrl.throttle = speed_throttle(vehicle, CRUISE_SPEED)
        ctrl.brake    = 0.0
        ctrl.steer    = steer_sign * steer_mag
        vehicle.apply_control(ctrl)
        tick_world(world, vehicle, spectator)


def stabilise_phase(world, vehicle, spectator, original_lane_id, direction, n_ticks):
    """Decision T+1 — UNCHANGED, traffic-blind on purpose."""
    for _ in range(n_ticks):
        ctrl          = carla.VehicleControl()
        ctrl.throttle = speed_throttle(vehicle, CRUISE_SPEED)
        ctrl.brake    = 0.0
        ctrl.steer    = target_lane_steer(vehicle, world, original_lane_id,
                                          direction, CRUISE_LOOKAHEAD)
        vehicle.apply_control(ctrl)
        tick_world(world, vehicle, spectator)


# ── Escort NPC: per-pair, target-lane-matched ────────────────────────────────

def escort_spawn_transform(target_wp, dist, ahead):
    """A transform on the TARGET lane's own centreline, `dist` metres ahead of
    (or behind) `target_wp`. Using the lane's waypoint chain rather than a
    nearby map spawn point guarantees the escort is correctly positioned AND
    correctly oriented for that lane, including through curves."""
    cands = target_wp.next(dist) if ahead else target_wp.previous(dist)
    if not cands:
        return None
    tf = cands[0].transform
    return carla.Transform(
        carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + 0.3),
        tf.rotation)


def spawn_pair_escort(world, tm, blueprint_library, vehicle,
                      original_lane_id, direction, rng):
    """
    Place one escort in the resolved target lane at a controlled longitudinal
    separation (see ESCORT_* config for the rewrite rationale). Returns
    (escorts, signed_offset_m) — the offset is logged per pair so any
    remaining crash can be audited against how far away the escort actually
    was, instead of being argued about.

    The escort runs under Traffic Manager with auto_lane_change DISABLED.
    That gives proper lane-following through curves and TM's own collision
    avoidance, and it fixes the previous approach's real defect: a one-shot
    apply_control() is never re-applied per tick (leaving a frozen throttle),
    and enable_constant_velocity() ignores physics entirely so the escort
    drove dead straight while the road curved — drifting into the ego's lane.
    TM handles both correctly. auto_lane_change(False) is what makes TM safe
    to use here: without it TM could move the escort out of the target lane
    mid-pair, which is the reason the first version avoided TM at all.

    Approach B note: the escort reacting to the ego (TM braking/avoidance) is
    fine and desirable — realistic traffic. What must never happen is the
    EGO's scripted decision reacting to the escort, and it does not: severity
    is sampled and applied on a fixed schedule, traffic-blind, unchanged.
    """
    if not ESCORT_ENABLED:
        return [], None   # see ESCORT_ENABLED rationale in the config block

    target_wp, _ = resolve_target_waypoint(world, vehicle, original_lane_id, direction)
    if target_wp is None:
        return [], None

    dist  = round(rng.uniform(*ESCORT_SPAWN_DIST_RANGE), 1)
    ahead = rng.random() < ESCORT_AHEAD_PROB
    spawn_tf = escort_spawn_transform(target_wp, dist, ahead)
    if spawn_tf is None:
        return [], None

    # Refuse to stack onto anything already there (ambient NPC or the ego).
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.get_transform().location.distance(spawn_tf.location) < ESCORT_CLEARANCE:
            return [], None

    bps = [bp for bp in blueprint_library.filter('vehicle.*')
           if int(bp.get_attribute('number_of_wheels')) == 4]
    npc = world.try_spawn_actor(rng.choice(bps), spawn_tf)
    if npc is None:
        return [], None

    # Give the escort an initial velocity matching traffic speed BEFORE handing
    # it to TM. Fixed 2026-07-22 (third pilot): a freshly spawned actor starts
    # STATIONARY, and TM has to accelerate it from 0. Over a pair's ~1.5 s the
    # escort covers only ~3-5 m while an ego already cruising at CRUISE_SPEED
    # covers 12-16 m — so the ego consumes the entire 10-16 m lead gap and
    # rear-ends what is effectively a parked car in the target lane.
    # The pilot data called this out unambiguously: 18/19 crashes had the
    # escort AHEAD (78% crash rate) vs 1/24 with it BEHIND (4%), and crash
    # pairs averaged speed_t 6.35 m/s vs 7.55 on safe pairs (already braking
    # into the impact). A stationary-ahead escort is a spawning artifact, not
    # a driving conflict, so crashes it causes are not legitimate outcomes.
    fwd = npc.get_transform().get_forward_vector()
    npc.set_target_velocity(carla.Vector3D(
        x=fwd.x * CRUISE_SPEED, y=fwd.y * CRUISE_SPEED, z=0.0))
    npc.set_autopilot(True, TM_PORT)
    tm.auto_lane_change(npc, False)
    tm.vehicle_percentage_speed_difference(npc, ESCORT_SPEED_PCT)
    return [npc], round(dist if ahead else -dist, 1)


def destroy_escorts(escorts):
    for npc in escorts:
        try:
            npc.set_autopilot(False, TM_PORT)
            npc.destroy()
        except Exception:
            pass


# ── Ambient NPC pool (ESCORT above does the RSS-coverage work; this pool is
# background realism, same role as in scenario_c_npc_rss.py) ────────────────

def spawn_ambient_npcs(world, client, blueprint_library, all_spawns, ego_spawns):
    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(TM_SEED)
    bps = [bp for bp in blueprint_library.filter('vehicle.*')
           if int(bp.get_attribute('number_of_wheels')) == 4]

    ego_spawn_ids = {id(sp) for sp in ego_spawns}
    candidates = [sp for sp in all_spawns if id(sp) not in ego_spawn_ids]

    def is_near_ego(sp):
        return any(sp.location.distance(e.location) <= NPC_SPAWN_RADIUS
                   for e in ego_spawns)

    near_flags = [is_near_ego(sp) for sp in candidates]
    near = [sp for sp, f in zip(candidates, near_flags) if f]
    far  = [sp for sp, f in zip(candidates, near_flags) if not f]
    npc_rng = random.Random(NPC_RNG_SEED)
    npc_rng.shuffle(near)
    npc_rng.shuffle(far)

    npcs = []
    for sp in near + far:
        if len(npcs) >= N_NPC:
            break
        npc = world.try_spawn_actor(npc_rng.choice(bps), sp)
        if npc is not None:
            npc.set_autopilot(True, TM_PORT)
            npcs.append(npc)
    print(f"  Ambient NPC pool: {len(npcs)}/{N_NPC} spawned "
          f"({len(near)} near-ego candidates available, {len(far)} far)")
    return tm, npcs


# ── Spawn filtering — UNCHANGED from banked script ───────────────────────────

def filter_spawn_points(world, all_spawns):
    good = []
    for sp in all_spawns:
        wp = world.get_map().get_waypoint(
            sp.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if wp is None:
            continue
        if adjacent_lane(wp, 'left') is not None or adjacent_lane(wp, 'right') is not None:
            good.append(sp)
    return good


# ── Episode loop ──────────────────────────────────────────────────────────────

def run_episode(episode_id, vehicle, world, spectator, writer, discard_writer,
                collision_state, sev_rng, escort_rng, spawn_point,
                blueprint_library, all_spawns, tm):
    pairs_logged      = 0
    no_lane_fails     = 0
    crashes           = 0
    off_roads         = 0
    wrong_ways        = 0
    no_clear_starts   = 0
    escorts_used      = 0
    severity_values   = []
    lat_dev_t_values  = []
    component_values  = []
    completed_flags   = []
    crash_escort_longi = []   # signed offset of the escort on crash pairs
    crash_no_escort    = 0    # crash pairs where no escort was placed at all
    escort_outcomes    = []   # (signed_offset, crashed?) for EVERY escorted
                              # pair — the ahead/behind split is what exposed
                              # the stationary-spawn bug, so it's now a
                              # standing gate rather than an ad-hoc CSV pull
    direction         = 'left'

    def log_discard(pair_attempt, failure_type, severity=None):
        transform = vehicle.get_transform()
        velocity  = vehicle.get_velocity()
        discard_writer.writerow({
            'episode_id'  : episode_id,
            'pair_attempt': pair_attempt,
            'failure_type': failure_type,
            'severity_t'  : severity,
            'spawn_x'     : round(spawn_point.location.x, 4),
            'spawn_y'     : round(spawn_point.location.y, 4),
            'fail_x'      : round(transform.location.x, 4),
            'fail_y'      : round(transform.location.y, 4),
            'fail_speed'  : round(math.sqrt(
                velocity.x**2 + velocity.y**2 + velocity.z**2), 4),
            'hit_actor'   : collision_state.get('last_actor'),
        })

    for pair in range(PAIRS_PER_EPISODE):
        placed = False
        for attempt in range(MAX_RELOCATE_TRIES):
            wp = world.get_map().get_waypoint(
                vehicle.get_transform().location,
                project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if wp is not None:
                if adjacent_lane(wp, direction) is not None:
                    placed = True
                    break
                other = 'right' if direction == 'left' else 'left'
                if adjacent_lane(wp, other) is not None:
                    direction = other
                    placed = True
                    break
            cruise(world, vehicle, spectator, RELOCATE_TICKS)
        if not placed:
            no_lane_fails += 1
            log_discard(pair, 'no_valid_lane')
            cruise(world, vehicle, spectator, RELOCATE_TICKS)
            continue

        wp_start         = world.get_map().get_waypoint(
            vehicle.get_transform().location,
            project_to_road=True, lane_type=carla.LaneType.Driving
        )
        original_lane_id = wp_start.lane_id
        target_lane_id   = adjacent_lane(wp_start, direction).lane_id

        # ── Clean-start precondition (transit-phase gate, not a decision
        # gate — see start_is_clear docstring) ────────────────────────────
        if not wait_for_clear_start(world, vehicle, spectator):
            no_clear_starts += 1
            log_discard(pair, 'no_clear_start')
            continue

        # ── Per-pair escort: spawned in the resolved target lane, right
        # before the decision — never gates the decision itself ───────────
        escorts, escort_longi = spawn_pair_escort(
            world, tm, blueprint_library, vehicle,
            original_lane_id, direction, escort_rng)
        if escorts:
            escorts_used += 1
        for _ in range(5):   # brief settle before the decision starts
            tick_world(world, vehicle, spectator)

        steer_mag   = round(sev_rng.uniform(*STEER_MAG_RANGE), 3)
        steer_ticks = sev_rng.randint(*STEER_TICKS_RANGE)
        severity    = round(steer_mag * steer_ticks, 4)
        collision_state['hit'] = False

        execute_steer_phase(world, vehicle, spectator, steer_mag, steer_ticks, direction)
        s_t    = get_state(vehicle, world)
        comp_t = compute_risk_components(vehicle, world, original_lane_id, direction)
        rss_t  = rss_lateral_snapshot(world, vehicle, original_lane_id, direction)

        stabilise_phase(world, vehicle, spectator, original_lane_id, direction,
                        TICKS_STABILISE_SAMPLE)
        s_t1    = get_state(vehicle, world)
        comp_t1 = compute_risk_components(vehicle, world, original_lane_id, direction)
        rss_t1  = rss_lateral_snapshot(world, vehicle, original_lane_id, direction)
        if rss_t1['present']:
            rss_present_count[0] += 1

        destroy_escorts(escorts)   # done with this pair's target lane

        completed = (s_t1['lane_id'] == target_lane_id)
        off_road  = comp_t1['lat_dev_m'] > 5.0
        wrong_way = is_wrong_way(world, vehicle)
        if collision_state['hit']:
            outcome = 'crash'
        elif wrong_way:
            outcome = 'wrong_way'
        elif off_road:
            outcome = 'off_road'
        else:
            outcome = 'safe'

        row = {
            'episode_id'    : episode_id, 't': pairs_logged,
            'x_t': s_t['x'], 'y_t': s_t['y'], 'speed_t': s_t['speed_ms'],
            'yaw_t': s_t['yaw'], 'lane_id_t': s_t['lane_id'], 'road_id_t': s_t['road_id'],
            'action_t': 'lane_change_' + direction,
            'steer_mag_t': steer_mag, 'steer_ticks_t': steer_ticks, 'severity_t': severity,
            'lat_dev_t': comp_t['lat_dev'], 'heading_err_t': comp_t['heading_err'],
            'speed_risk_t': comp_t['speed_risk'],
            'lat_dev_m_t': comp_t['lat_dev_m'], 'heading_deg_t': comp_t['heading_deg'],
            'rho_t': comp_t['rho'],
            'x_t1': s_t1['x'], 'y_t1': s_t1['y'], 'speed_t1': s_t1['speed_ms'],
            'yaw_t1': s_t1['yaw'], 'lane_id_t1': s_t1['lane_id'], 'road_id_t1': s_t1['road_id'],
            'lat_dev_t1': comp_t1['lat_dev'], 'heading_err_t1': comp_t1['heading_err'],
            'speed_risk_t1': comp_t1['speed_risk'],
            'lat_dev_m_t1': comp_t1['lat_dev_m'], 'heading_deg_t1': comp_t1['heading_deg'],
            'lane_change_completed': completed,
            'escort_longi_t': escort_longi,   # signed m; +ahead / -behind /
                                              # blank when no escort was placed
            'rho_actual': comp_t1['rho'], 'rho_baseline': None, 'RPI_pair': None,
            'scenario': SCENARIO_TYPE, 'weather': WEATHER, 'npc_density': N_NPC,
            'outcome': outcome,
        }
        for prefix, snap in (('rss_t_', rss_t), ('rss_t1_', rss_t1)):
            for k, v in snap.items():
                row[prefix + k] = v
        writer.writerow(row)

        pairs_logged += 1
        if escort_longi is not None:
            escort_outcomes.append((escort_longi, outcome == 'crash'))
        severity_values.append(severity)
        lat_dev_t_values.append(comp_t['lat_dev_m'])
        component_values.append(comp_t1)
        completed_flags.append(completed)

        if outcome == 'crash':
            crashes += 1
            if escort_longi is None:
                crash_no_escort += 1
            else:
                crash_escort_longi.append(escort_longi)
            log_discard(pair, 'crash_ends_episode', severity)
            break
        if outcome == 'off_road':
            off_roads += 1
            log_discard(pair, 'off_road_ends_episode', severity)
            break
        if outcome == 'wrong_way':
            # Ego is on an opposing lane. The pair IS logged (it is a genuine
            # high-risk outcome of an overshooting change, and discarding it
            # would create the severity-correlated hole design rule 8 warns
            # about), but the episode ends — every later pair would be
            # measured from an invalid heading.
            wrong_ways += 1
            log_discard(pair, 'wrong_way_ends_episode', severity)
            break

        cruise(world, vehicle, spectator, TICKS_SETTLE)
        direction = 'right' if direction == 'left' else 'left'

    return (pairs_logged, no_lane_fails, crashes, off_roads, escorts_used,
            severity_values, lat_dev_t_values, component_values, completed_flags,
            crash_escort_longi, crash_no_escort, escort_outcomes,
            wrong_ways, no_clear_starts)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT)
    world  = client.get_world()
    print(f"Connected. Map: {world.get_map().name}")
    print(f"PILOT_MODE = {PILOT_MODE}  (NUM_EPISODES = {NUM_EPISODES}, "
          f"ambient NPCs = {N_NPC})")

    settings                     = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp        = blueprint_library.filter('vehicle.tesla.model3')[0]
    vehicle_bp.set_attribute('color', '200,120,0')
    collision_bp      = blueprint_library.find('sensor.other.collision')

    all_spawns = world.get_map().get_spawn_points()
    ego_spawns = filter_spawn_points(world, all_spawns)
    print(f"Spawn filter: {len(ego_spawns)}/{len(all_spawns)} points have a "
          f"valid same-direction adjacent lane")
    if not ego_spawns:
        raise RuntimeError("No lane-change-capable spawn points on this map.")
    spectator = world.get_spectator()

    tm, npcs = spawn_ambient_npcs(world, client, blueprint_library, all_spawns, ego_spawns)
    for _ in range(40):
        world.tick()

    base_fields = [
        'episode_id', 't',
        'x_t', 'y_t', 'speed_t', 'yaw_t', 'lane_id_t', 'road_id_t',
        'action_t', 'steer_mag_t', 'steer_ticks_t', 'severity_t',
        'lat_dev_t', 'heading_err_t', 'speed_risk_t',
        'lat_dev_m_t', 'heading_deg_t', 'rho_t',
        'x_t1', 'y_t1', 'speed_t1', 'yaw_t1', 'lane_id_t1', 'road_id_t1',
        'lat_dev_t1', 'heading_err_t1', 'speed_risk_t1',
        'lat_dev_m_t1', 'heading_deg_t1',
        'lane_change_completed', 'escort_longi_t',
        'rho_actual', 'rho_baseline', 'RPI_pair',
        'scenario', 'weather', 'npc_density', 'outcome',
    ]
    rss_fields = [p + k for p in ('rss_t_', 'rss_t1_')
                  for k in ('present', 'gap', 'ego_lat_v', 'npc_lat_v', 'margin',
                            'longi', 'long_gap', 'long_margin', 'rss_dangerous')]
    fieldnames = base_fields + rss_fields
    discard_fieldnames = [
        'episode_id', 'pair_attempt', 'failure_type', 'severity_t',
        'spawn_x', 'spawn_y', 'fail_x', 'fail_y', 'fail_speed', 'hit_actor',
    ]
    episode_stats_fieldnames = [
        'episode_id', 'spawn_x', 'spawn_y', 'pairs_logged',
        'no_lane_fails', 'crashes', 'off_roads', 'completed_changes',
        'escorts_used', 'status',
    ]

    sev_rng    = random.Random(RNG_SEED)
    escort_rng = random.Random(ESCORT_RNG_SEED)

    total_pairs, total_no_lane, total_crashes, total_off_roads = 0, 0, 0, 0
    total_wrong_ways, total_no_clear = 0, 0
    failed_episodes = []
    all_severity, all_lat_dev_t, all_components, all_completed = [], [], [], []
    crash_escort_longi = []   # run-level crash attribution (see Gate 4)
    crash_no_escort    = 0
    escort_outcomes    = []
    all_rows_for_gates = []   # (episode_id, rss_t1_present dicts) for coverage gates

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
                spawn_point      = ego_spawns[ep % len(ego_spawns)]
                vehicle          = None
                collision_sensor = None
                collision_state  = {'hit': False, 'total_hits': 0, 'last_actor': None}

                def on_collision(event):
                    collision_state['hit'] = True
                    collision_state['total_hits'] += 1
                    # Record WHAT was hit (2026-07-22). The previous
                    # diagnostic only recorded that an escort existed on the
                    # pair, which cannot distinguish "hit the escort" from
                    # "hit an ambient NPC / a wall" — that ambiguity cost a
                    # full pilot round. event.other_actor is authoritative.
                    try:
                        collision_state['last_actor'] = event.other_actor.type_id
                    except Exception:
                        collision_state['last_actor'] = 'unknown'

                try:
                    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
                    if vehicle is None:
                        for _ in range(40):
                            world.tick()
                        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                    collision_sensor = world.spawn_actor(
                        collision_bp, carla.Transform(), attach_to=vehicle)
                    collision_sensor.listen(on_collision)

                    print(f"Episode {ep+1}/{NUM_EPISODES} | "
                          f"Spawn: ({spawn_point.location.x:.1f}, "
                          f"{spawn_point.location.y:.1f})", flush=True)

                    cruise(world, vehicle, spectator, WARMUP_TICKS)

                    (pairs, no_lane, crashes, off_roads, escorts_used, sev, latm,
                     comps, completed, ep_crash_longi, ep_crash_none,
                     ep_escort_outcomes, ep_wrong_ways, ep_no_clear) = run_episode(
                        ep + 1, vehicle, world, spectator, writer, discard_writer,
                        collision_state, sev_rng, escort_rng, spawn_point,
                        blueprint_library, all_spawns, tm
                    )
                    total_pairs     += pairs
                    total_no_lane   += no_lane
                    total_crashes   += crashes
                    total_off_roads += off_roads
                    all_severity.extend(sev)
                    all_lat_dev_t.extend(latm)
                    all_components.extend(comps)
                    all_completed.extend(completed)
                    crash_escort_longi.extend(ep_crash_longi)
                    crash_no_escort += ep_crash_none
                    escort_outcomes.extend(ep_escort_outcomes)
                    total_wrong_ways += ep_wrong_ways
                    total_no_clear   += ep_no_clear

                    stats_writer.writerow({
                        'episode_id': ep + 1,
                        'spawn_x': round(spawn_point.location.x, 4),
                        'spawn_y': round(spawn_point.location.y, 4),
                        'pairs_logged': pairs, 'no_lane_fails': no_lane,
                        'crashes': crashes, 'off_roads': off_roads,
                        'completed_changes': sum(completed),
                        'escorts_used': escorts_used, 'status': 'ok',
                    })
                    print(f"  Episode {ep+1} complete | pairs: {pairs} | "
                          f"escorts used: {escorts_used}/{pairs} | "
                          f"completed changes: {sum(completed)}/{pairs} | "
                          f"crashes: {crashes} | off_roads: {off_roads}", flush=True)

                except Exception as e:
                    error_msg = f"Episode {ep+1} FAILED: {type(e).__name__}: {e}"
                    print(f"  {error_msg}", flush=True)
                    log_error(error_msg)
                    failed_episodes.append(ep + 1)
                    stats_writer.writerow({
                        'episode_id': ep + 1,
                        'spawn_x': round(spawn_point.location.x, 4),
                        'spawn_y': round(spawn_point.location.y, 4),
                        'pairs_logged': None, 'no_lane_fails': None,
                        'crashes': None, 'off_roads': None,
                        'completed_changes': None, 'escorts_used': None,
                        'status': 'exception',
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
                    for f in (csvfile, discardfile, statsfile):
                        f.flush()
                        os.fsync(f.fileno())

        max_possible = NUM_EPISODES * PAIRS_PER_EPISODE
        print(f"\nAll done. Total pairs: {total_pairs} / {max_possible} possible "
              f"({100*total_pairs/max_possible:.1f}%)")
        print(f"  no_lane_fails: {total_no_lane} | crashes: {total_crashes} | "
              f"off_roads: {total_off_roads} | wrong_way: {total_wrong_ways} | "
              f"no_clear_start: {total_no_clear} | "
              f"completed changes: {sum(all_completed)}/{total_pairs}")
        print("  (wrong_way counts episodes ended because the ego was left on an")
        print("   opposing lane — previously these produced a full-lock U-turn")
        print("   into oncoming traffic and were miscounted as ordinary crashes.)")

        # ── Gate 1: severity spread + severity->T-state link (unchanged design) ──
        if all_severity:
            sev_std = statistics.pstdev(all_severity) if len(all_severity) > 1 else 0.0
            lat_std = statistics.pstdev(all_lat_dev_t) if len(all_lat_dev_t) > 1 else 0.0
            r_sev_lat = pearson_r(all_severity, all_lat_dev_t)
            print(f"\n── Decision-T severity gate ({len(all_severity)} pairs) ──")
            print(f"  severity std {sev_std:.3f} | lat_dev@T std {lat_std:.3f} | "
                  f"corr(severity, lat_dev@T) = {r_sev_lat:+.3f}")
            log_error(f"SEVERITY GATE: sev_std={sev_std:.4f} lat_std={lat_std:.4f} r={r_sev_lat:+.4f}")

        # ── Gate 2: per-component variance at T+1 ────────────────────────────
        if all_components:
            print(f"\n── Per-component variance check ({len(all_components)} pairs, T+1) ──")
            for key in ('lat_dev', 'heading_err', 'speed_risk'):
                vals = [c[key] for c in all_components]
                mean_v = statistics.mean(vals)
                std_v  = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                near_cap = sum(1 for v in vals if v > 0.95) / len(vals)
                print(f"  {key:12s} mean={mean_v:.4f} std={std_v:.4f} frac>0.95={near_cap:.2f}")
                log_error(f"COMPONENT CHECK {key}: mean={mean_v:.4f} std={std_v:.4f} frac_cap={near_cap:.3f}")

        # ── Gate 3: completion mix ────────────────────────────────────────────
        if all_completed:
            frac = sum(all_completed) / len(all_completed)
            print(f"\n── Completion mix ── {sum(all_completed)}/{len(all_completed)} "
                  f"changes completed ({100*frac:.0f}%)")
            log_error(f"COMPLETION MIX: {sum(all_completed)}/{len(all_completed)}")

        # ── Gate 3b: RSS coverage (added 2026-07-22) ─────────────────────────
        # Coverage is the detector for "did the setup accidentally sanitize the
        # target lane". The escort-disabled pilot hit 1.9% and nobody would
        # have known from the console without this gate — the run looked
        # healthy on every other metric while measuring merges into empty
        # lanes. Low coverage does NOT invalidate the rho ablation, but it does
        # mean this run cannot support an RSS baseline row for Scenario A, and
        # it weakens any claim that the decisions happened "in traffic".
        rss_present_t1 = rss_present_count[0]
        if all_completed:
            cov = rss_present_t1 / max(len(all_completed), 1)
            print(f"\n── Gate 3b: RSS coverage at T+1 ──")
            print(f"  target-lane vehicle present in {rss_present_t1}/{len(all_completed)} "
                  f"pairs ({100*cov:.1f}%)")
            print("  GATE: coverage in the low single digits means decisions are")
            print("  happening in an empty target lane — check CLEAR_START_RADIUS /")
            print("  CLEAR_START_LATERAL before raising N_NPC, since an over-wide")
            print("  clean-start band excludes traffic far more effectively than")
            print("  extra NPCs can add it.")
            log_error(f"RSS COVERAGE t1: {rss_present_t1}/{len(all_completed)} = {cov:.4f}")

        # ── Gate 4: crash attribution (2026-07-22) ───────────────────────────
        # The second pilot's 16/20-episode crash rate was traced to escorts
        # being spawned 0-2 m from the ego. This gate makes that failure mode
        # self-reporting instead of requiring a manual CSV pull: if crashes
        # are still frequent AND concentrated at small |escort_longi_t|, the
        # separation range is still too tight. If crashes are spread across
        # the full range (or happen with no escort at all), they're genuine
        # merge failures and the range is fine.
        if crash_escort_longi or crash_no_escort:
            n_cr = len(crash_escort_longi) + crash_no_escort
            print(f"\n── Crash attribution ({n_cr} crash pairs) ──")
            print(f"  with an escort placed: {len(crash_escort_longi)} | "
                  f"with no escort: {crash_no_escort}")
            if crash_escort_longi:
                absd = [abs(d) for d in crash_escort_longi]
                print(f"  |escort offset| at crash: mean {statistics.mean(absd):.1f} m, "
                      f"min {min(absd):.1f} m, max {max(absd):.1f} m "
                      f"(spawn range was {ESCORT_SPAWN_DIST_RANGE[0]:.0f}-"
                      f"{ESCORT_SPAWN_DIST_RANGE[1]:.0f} m)")
            log_error(f"CRASH ATTRIBUTION: with_escort={len(crash_escort_longi)} "
                      f"no_escort={crash_no_escort}")

        # Ahead/behind split — this is the signal that exposed the
        # stationary-spawn bug (18/19 crashes with the escort AHEAD vs 1/24
        # BEHIND). A large asymmetry here means the escort is still behaving
        # as an obstacle rather than as traffic; roughly symmetric rates mean
        # crashes are genuine merge conflicts.
        if escort_outcomes:
            ah = [c for d, c in escort_outcomes if d > 0]
            bh = [c for d, c in escort_outcomes if d < 0]
            print(f"\n── Escort ahead/behind crash split ──")
            if ah:
                print(f"  escort AHEAD : {sum(ah)}/{len(ah)} crashed "
                      f"({100*sum(ah)/len(ah):.0f}%)")
            if bh:
                print(f"  escort BEHIND: {sum(bh)}/{len(bh)} crashed "
                      f"({100*sum(bh)/len(bh):.0f}%)")
            print("  GATE: a large ahead-vs-behind gap means the escort is")
            print("  still acting as an obstacle (e.g. spawning slower than")
            print("  traffic), not as a merge conflict. Comparable rates are")
            print("  what a physically fair setup looks like.")
            log_error(f"ESCORT SPLIT: ahead={sum(ah)}/{len(ah)} behind={sum(bh)}/{len(bh)}")

        # ── Gate 5: RSS lateral coverage + heading diagnostic ────────────────
        if heading_stats['checked']:
            hr = heading_stats['rejected'] / heading_stats['checked']
            print(f"\n── Heading-mismatch rejections ──")
            print(f"  {heading_stats['checked']} target-lane candidates checked, "
                  f"{heading_stats['rejected']} rejected as wrong-direction "
                  f"({100*hr:.1f}%)")
            log_error(f"HEADING: checked={heading_stats['checked']} "
                      f"rejected={heading_stats['rejected']} rate={hr:.3f}")

        if failed_episodes:
            summary = f"{len(failed_episodes)}/{NUM_EPISODES} episodes failed: {failed_episodes}"
            print(summary, flush=True)
            log_error(f"RUN SUMMARY: {summary}")
        else:
            log_error(f"RUN SUMMARY: all {NUM_EPISODES} episodes completed cleanly.")

        print(f"\nData saved to: {OUTPUT_FILE}")
        print(f"Discarded-attempt log: {DISCARD_LOG_FILE}")
        print(f"Per-episode stats: {EPISODE_STATS_FILE}")
        print("NOTE: RSS pair-level and per-episode coverage percentages are")
        print("printed per-episode above (escorts used/pairs) — the same")
        print("Gate-4-style per-episode distribution check used for Scenario C")
        print("should be run on the CSV before trusting any aggregate coverage")
        print("number; see analyse_scenario_c_npc_rss.py's coverage_report()")
        print("pattern for scenario_a_lanechange_npc_rss.csv once this exists.")

    finally:
        print("Cleaning up ambient NPCs and restoring async mode...")
        for npc in npcs:
            try:
                npc.set_autopilot(False, TM_PORT)
                npc.destroy()
            except Exception:
                pass
        for _ in range(10):
            world.tick()
        settings                  = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
        print("Done.")


if __name__ == '__main__':
    main()
