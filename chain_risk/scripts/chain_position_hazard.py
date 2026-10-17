"""
Does a decision's crash risk depend on its position in a sequence?

Implements the design frozen in ../docs/PREREGISTRATION.md, including AMENDMENT 1. Read that file
before changing anything here.

WHY THIS STUDY EXISTS (and how it differs from the closed RPI project):
The previous project measured a risk score computed purely from the vehicle state at the sampling
instant, while the treatment deterministically set that state. The correlation it found was therefore
arithmetic, and no possible result could have been null. Here the outcome variable is a COLLISION
recorded by CARLA's collision sensor. It is not a formula we wrote, it is not a deterministic function
of the treatment, and a flat hazard across positions is a live and reportable outcome.

THE DESIGN — three arms, two of which cost nothing extra:

  CHAIN      8 scripted brake decisions per episode. Hazard measured at each position t=0..7.
  ISO-EARLY  IS CHAIN position 0 — fresh spawn, warmup, zero prior decisions. Free baseline.
  ISO-LATE   Cruise for a duration matched to 7 decision cycles, then make exactly ONE decision.

  PRIMARY:   hazard(CHAIN t=7)  vs  hazard(ISO-LATE)
  SECONDARY: hazard trend across CHAIN t=0..7

WHY ISO-LATE IS THE WHOLE EXPERIMENT: by position 7 the ego is ~150 m from spawn, in different road
geometry. Without ISO-LATE a rising hazard could just mean "the road got worse". ISO-LATE matches
elapsed time and distance while holding prior-decision count at ZERO. This is the control the RPI
project never had — its Scenario D was declared a control on the grounds that no ablation was defined
for it, so was never tested; when it finally was, it scored r=+0.868 with no treatment applied.

ARMS ALTERNATE WITHIN ONE RUN (ep 0 CHAIN, ep 1 ISO-LATE, ...) so session-level drift — server state,
machine load, thermal throttling — cannot confound the primary comparison.

THE FOLLOWER (rewritten per AMENDMENT 1):
Pilot 1 produced 0 crashes in 90 decisions with a Traffic-Manager follower. Root cause: TM has
effectively zero reaction delay and near-perfect collision avoidance — it brakes the same tick the ego
does. No gap setting produces rear-end collisions against a driver with no perception lag.

The follower is therefore implemented HERE, explicitly, as a linear car-following controller acting on
a DELAYED observation of the ego: it responds to where the ego was FOLLOWER_REACTION_TICKS ago, not
where it is now. Two benefits beyond fixing the crash rate:
  1. Results characterise a documented model we specify, not CARLA's opaque internal traffic model.
  2. Collisions have an interpretable cause — the follower reacted tau seconds late and the gap was
     insufficient — rather than emerging from a black box.

Lessons carried over from the previous project's escort work, do not regress these:
  - follower spawned ONCE PER EPISODE and persists. Re-spawning per decision would reset exactly the
    accumulated state this study measures.
  - given set_target_velocity() at spawn; a fresh actor starts at 0 m/s and behaves as a parked
    obstacle otherwise.
  - do NOT use enable_constant_velocity() — it ignores physics and drives straight through curves.

Run:  py -3.7 chain_risk\\scripts\\chain_position_hazard.py
"""
import carla
import math
import csv
import os
import random
import datetime
import statistics
from collections import deque, Counter

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOST, PORT, TIMEOUT = 'localhost', 2000, 30.0

PILOT_MODE       = False
# Per arm, PER SWEEP SETTING. 40 x 2 arms x 4 settings = 320 episodes.
# Sized from the v1 sweep's yield: roughly half of logged pairs produce a valid following-gap
# reading, and only episodes surviving to position 7 contribute to the primary comparison, so
# 10 episodes/arm gave n=5 there — too few for an interval. 40 should give n~20 per setting.
EPISODES_PER_ARM = 10 if PILOT_MODE else 40

# ── AMENDMENT 10 — INTERVAL SWEEP. The null becomes a dose-response. ─────────
# Pilot 9 measured that 89% of a braking disturbance is erased before the next decision, over an
# inter-decision interval of 2.90 s. Fitting exponential relaxation gives a follower time constant
#
#     tau = 2.90 / ln(1/0.11) = 1.31 s
#
# which explains the null completely: decisions were spaced at 2.2*tau, so the follower returned to
# equilibrium every time and nothing could accumulate. That is a measurement, not an absence.
#
# It also makes a QUANTITATIVE, FALSIFIABLE prediction, pre-registered here BEFORE the run:
# persistence of a disturbance into the next decision should follow exp(-interval/tau).
#
#   TICKS_CRUISE  interval    /tau    PREDICTED persistence   predicted steady-state amplification
#        0         0.90 s     0.69           0.50                        2.00x
#       12         1.50 s     1.15           0.32                        1.47x
#       40         2.90 s     2.21           0.11                        1.12x   <- pilot 9, measured
#       76         4.70 s     3.59           0.03                        1.03x
#
# The 2.90 s row is pilot 9's measured 0.11, so the curve is anchored to existing data at one point
# and extrapolates to three untested ones. If persistence follows the predicted curve, carry-forward
# risk is governed by a measurable relaxation constant rather than by decision count — a positive,
# conditional result. If it does not, the exponential-relaxation model is wrong and that is also
# informative.
#
# ONLY ONE VARIABLE CHANGES. Brake severity range, follower model, entry-speed gating and every
# other constant are byte-identical to pilot 9. This project has repeatedly changed two things at
# once and then been unable to attribute the result; not here.
SWEEP_CRUISE_TICKS = [0, 12, 40, 76]
TAU_ESTIMATE_S     = 1.31     # from pilot 9; the sweep tests it, it is not assumed
# ──────────────────────────────────────────────────────────────────────────────

DECISIONS_PER_CHAIN = 8
TICKS_CRUISE        = 40
TICKS_RECOVER       = 10

# ── AMENDMENT 9 — the chain was decaying into a regime where crashes are impossible ──
# Pilot 8 passed GATE 1 (10.9%) but the per-position breakdown showed crashes only at t=0,1,2 and
# zero at t=3..7. The cause was not chain position and not frailty selection
# (corr(episode length, t=0 gap) = 0.08 — episodes did not crash early because they started tight).
# It was ego SPEED DECAY across the chain:
#
#   position:        t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7     ISO-LATE
#   follower speed: 7.69   3.41   1.63   3.61   1.49   1.76   1.25   0.00       11.03
#   gap at decision 12.2   12.5   16.9   22.1   14.7   13.6    9.7    5.0       23.0
#
# TICKS_RECOVER=10 (0.5 s) was inherited from RPI Scenario C, where it was deliberately short so
# that T+1 sampled a live transient. Here it has an entirely different consequence: after a 2.25 s
# stop-hold the ego gets 0.5 s of recovery plus 2.0 s at half throttle, which is nowhere near enough
# to return to ~11 m/s from a standstill. So each cycle starts slower than the last and by t=3 both
# vehicles are crawling. A rear-end collision needs closing kinetic energy; there is none left.
#
# This also invalidates the apparent headline. Gap falls 12.2 m -> 5.0 m across the chain, which
# looks like headway erosion, but the follower's DESIRED gap is 2.0 + 0.5*v, so it collapses with
# the speed. Normalised, the excess gap goes +6.3 m -> +3.0 m, and at t=7 both vehicles are
# stationary 5 m apart — a queue at rest, not a tailgating hazard. CHAIN t=7 (0.00 m/s) and
# ISO-LATE (11.03 m/s) were never in the same physical regime, so comparing them measured the speed
# difference, not the position effect. Time-matching matched duration; nothing matched speed.
#
# This is the SIXTH instance of the project's recurring bug class: the system drifts into a state
# where the outcome of interest is near-impossible, and the measurement then reports the drift.
#
# FIX: gate recovery on SPEED, not ticks. Every decision now begins from a comparable ego speed, so
# position is the only thing that varies systematically. This does NOT reset the follower — its
# position, speed and accumulated disturbance carry forward untouched, which is precisely the
# carry-forward the study is testing. The null remains fully available: if the follower re-
# establishes its equilibrium headway during each spin-up, t=7 will look exactly like t=0.
ENTRY_SPEED_TARGET  = 9.0    # m/s the ego must regain before a decision may begin
ENTRY_SPEED_TOL     = 1.0    # m/s band; decisions outside it are logged and excluded by gate
MAX_SPINUP_TICKS    = 150    # 7.5 s cap so a wedged ego cannot hang the episode
WARMUP_TICKS        = 50
CRUISE_THROTTLE     = 0.5

# ── AMENDMENT 8 — brake duration range shifted. This is a POWER change, not a bias change ──
# The original range was inherited from the banked RPI Scenario C, where SEVERITY WAS THE TREATMENT
# and a wide range was essential. In THIS study the treatment is POSITION IN THE SEQUENCE; severity
# is a nuisance variable. Measured consequence of keeping the old range: only 13% of decisions
# actually halt the ego (87% are taps of 0.25-0.6 s that a follower absorbs trivially), and a
# collision is only possible when the ego halts. 13% x 30% crash-given-halt = 3.9%, below the gate.
# 87% of samples were incapable of producing the outcome and so contributed nothing.
#
# Severity remains randomised INDEPENDENTLY of position (GATE 3 verifies it stays flat across
# t=0..7), so this cannot bias the position comparison — it only stops wasting samples.
# Duration 5-20 -> 10-22 ticks roughly doubles the halt rate to ~26%; at the measured 30%
# crash-given-halt that gives ~8%, inside the 5-15% gate, and P(reach t=7) = 0.92^7 = 56%, which
# keeps GATE 2 passing. Intensity range deliberately left untouched.
BRAKE_INTENSITY_RANGE = (0.15, 0.9)
BRAKE_DURATION_RANGE  = (10, 22)

# ── Follower model (AMENDMENT 1) ─────────────────────────────────────────────
# PRIMARY CRASH-RATE LEVER is now the reaction delay, not the gap. Human brake reaction time is
# roughly 1.0-1.5 s; at 0.05 s/tick that is 20-30 ticks. Only ONE tuning attempt remains under the
# pre-registered budget.
FOLLOWER_REACTION_TICKS = 25      # 1.25 s perception/reaction lag
FOLLOWER_GAP_RANGE      = (6.0, 12.0)   # m behind the ego at spawn
# AMENDMENT 3: blueprint PINNED. Previously rng.choice() over all 4-wheeled blueprints, which
# includes CARLA's heavy vehicles (firetruck, ambulance, carlacola). Measured consequence: episode 1's
# follower averaged 1.18 m/s while episode 19's averaged 8.06 m/s — half the followers physically
# could not keep station with a Model 3, so they drifted to 30-90 m and no collision was possible.
# Pinning to the ego's own model guarantees performance parity and removes a large noise source.
FOLLOWER_BLUEPRINT      = 'vehicle.tesla.model3'
# Tailgating parameters. Desired gap = MIN_GAP + HEADWAY * v, so at ~6 m/s this is ~6.6 m. With a
# 1.25 s lag the follower covers ~7.5 m before it even begins reacting, so a hard ego stop becomes a
# collision while a light brake does not — which is precisely the variation the hazard measure needs.
FOLLOWER_MIN_GAP        = 2.0     # m standstill gap
FOLLOWER_TIME_HEADWAY   = 0.5     # s desired headway (=> ~6 m desired at 8 m/s)

# ── AMENDMENT 6 — THE SCENARIO, not the follower, was the defect ─────────────
# Measured: during the HARDEST brakes the follower closes only 3.07 m on average (5.83 m max) while
# sitting 10.6 m back. A collision requires gap < closure, and it never is. The ego brake-pulses for
# 0.25-1.00 s then immediately accelerates away — shorter than the follower's 1.25 s reaction lag —
# so by the time a delayed reaction would matter the ego has already recovered. Amendments 1-5 all
# modified the perception model, which was never the binding constraint.
#
# Real rear-end collisions happen when a lead vehicle stops AND STAYS STOPPED. Adding a dwell at
# standstill turns this from a brake pulse into an emergency stop. The dwell only triggers when the
# brake actually brought the ego near a halt, so it is severity-dependent by construction: hard
# brakes produce a stopped obstacle, light brakes do not. That is exactly the variation the hazard
# measure needs, and it arises from the decision rather than being imposed.
# AMENDMENT 7: 30 -> 45 ticks (1.5 -> 2.25 s). Pilot 6 finally produced collisions and isolated the
# remaining shortfall exactly: 13% of decisions trigger the dwell, and 30% of THOSE crash, giving
# 13% x 30% = 3.9% overall against a 5% floor. The margin is razor-thin — gap was 6.6 m on crashes
# vs 7.1 m on non-crashes, half a metre apart — so a slightly longer stop converts most of the
# near-misses. Chosen over widening the brake-severity range (which would also work, since 79% of
# decisions are currently imperceptible taps) because THIS CHANGE CANNOT OVERSHOOT: crash rate is
# capped at the dwell-trigger rate of 13%, already inside the gate, even if every dwell became a
# collision. Widening severity has no such ceiling and could break Gate 2. It also leaves the
# a-priori severity lock untouched.
TICKS_STOP_HOLD    = 45      # 2.25 s dwell at standstill after a brake that nearly stopped the ego
STOP_HOLD_SPEED    = 2.0     # m/s — below this after braking, the ego holds the stop
FOLLOWER_K_GAP          = 0.35    # gap-error gain
FOLLOWER_K_SPEED        = 0.90    # speed-error gain
# BUGFIX 2026-09-19 (see PREREGISTRATION AMENDMENT 2): this was 7.0, which is BELOW the ego's
# observed cruise speed (max 11.43 m/s). The follower was hard-capped slower than the vehicle it
# was following, so it lost ground every cycle and drifted from ~19 m at t=0 to ~43 m at t=7 (87 m
# in ISO-LATE). At those distances no reaction delay can produce a collision — the road behind the
# ego was functionally empty, which is why pilots 1 and 2 both returned exactly zero crashes.
# Set high enough that it never binds; the car-following law's gap term is what limits approach
# speed, which is the intended control authority.
FOLLOWER_TARGET_SPEED   = 14.0    # m/s free-flow cap (deliberately non-binding)

TM_PORT, TM_SEED = 8000, 42

SEV_RNG_SEED   = 42   # decision severity ONLY
SPAWN_RNG_SEED = 43   # follower/spawn selection — separate so spawn mechanics can never perturb
                      # the severity draw sequence (lesson from the A variant)

OUTPUT_DIR  = r'C:\Users\jayad\Desktop\RPI_AINN\chain_risk\data'
# Identifies this run in the output filenames. Change it for every run worth keeping.
RUN_TAG     = 'sweep_v2'
OUTPUT_FILE = os.path.join(OUTPUT_DIR,
    # RUN-TAGGED OUTPUT (2026-10-03). This script previously wrote to a fixed filename, and the
    # Amendment 10 sweep silently overwrote the preceding pilot's data — the single-run null that
    # had already been reported survived only as prose and could not be recomputed or given a CI.
    # Output paths now carry a run tag, so a later run can never destroy an earlier one. Set
    # RUN_TAG before any run you intend to cite.
    'chain_position_hazard_%s%s.csv' % (RUN_TAG, '_pilot' if PILOT_MODE else ''))
EPISODE_STATS_FILE = OUTPUT_FILE.replace('.csv', '_episode_stats.csv')
ERROR_LOG_FILE     = OUTPUT_FILE.replace('.csv', '_errors.log')
# ──────────────────────────────────────────────────────────────────────────────


def log_error(msg):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(f"[{ts}] {msg}\n")


def _heading_diff_deg(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def speed_of(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def get_state(vehicle, world):
    tf = vehicle.get_transform()
    wp = world.get_map().get_waypoint(tf.location, project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
    return {
        'x': round(tf.location.x, 3), 'y': round(tf.location.y, 3),
        'speed': round(speed_of(vehicle), 4), 'yaw': round(tf.rotation.yaw, 3),
        'road_id': wp.road_id if wp else None,
        'lane_id': wp.lane_id if wp else None,
        'is_junction': bool(wp.is_junction) if wp else None,
    }


def apply_control(vehicle, throttle, brake, steer=0.0):
    c = carla.VehicleControl()
    c.throttle, c.brake, c.steer = throttle, brake, steer
    vehicle.apply_control(c)


def lane_direction_ok(wp, ref_yaw):
    """Reject a lane running against the vehicle's heading. Without this, an overshoot that snaps a
    vehicle onto an opposing lane makes next() point backwards and the steering controller commands
    a full-lock U-turn (diagnosed in the previous project's Scenario A)."""
    return _heading_diff_deg(wp.transform.rotation.yaw, ref_yaw) <= 90.0


def best_next_waypoint(wp, dist, ref_yaw):
    """next() can return several successors at a junction; taking [0] picks an arbitrary branch.
    Choose the best heading match, and reject the lot if even that one reverses."""
    cands = wp.next(dist)
    if not cands:
        return None
    best = min(cands, key=lambda c: _heading_diff_deg(c.transform.rotation.yaw, ref_yaw))
    return best if lane_direction_ok(best, ref_yaw) else None


def lane_follow_steer(vehicle, world, lookahead=6.0):
    tf = vehicle.get_transform()
    wp = world.get_map().get_waypoint(tf.location, project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
    if wp is None or not lane_direction_ok(wp, tf.rotation.yaw):
        return 0.0
    ahead = best_next_waypoint(wp, lookahead, tf.rotation.yaw)
    if ahead is None:
        return 0.0
    t = ahead.transform.location
    yaw_diff = math.atan2(t.y - tf.location.y, t.x - tf.location.x) - math.radians(tf.rotation.yaw)
    while yaw_diff >  math.pi: yaw_diff -= 2 * math.pi
    while yaw_diff < -math.pi: yaw_diff += 2 * math.pi
    return max(-1.0, min(1.0, yaw_diff / math.pi * 2.0))


def move_spectator(world, vehicle, spectator):
    """Chase camera.

    AMENDMENT 8: pulled back 12 m -> 30 m and raised 6 m -> 11 m. The follower rides 7-12 m behind
    the ego, so a 12 m chase camera placed it AT OR BEHIND the camera plane — it was essentially
    never in frame. Every visual check of this experiment showed a single car on an empty road, and
    the road genuinely looked empty even though the follower was present in the data throughout.
    Visual verification caught three real defects in this study that the printed diagnostics missed,
    so the camera needs to actually show both vehicles.
    """
    loc = vehicle.get_transform().location
    yaw = vehicle.get_transform().rotation.yaw
    fx, fy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    spectator.set_transform(carla.Transform(
        carla.Location(x=loc.x - 30 * fx, y=loc.y - 30 * fy, z=loc.z + 11),
        carla.Rotation(pitch=-18, yaw=yaw)))


# ── Follower with explicit reaction delay (AMENDMENT 1) ──────────────────────

class Follower:
    """Linear car-following controller acting on a DELAYED observation of the ego.

    Control law, applied every tick to the perceived (lagged) leader state:
        desired_gap   = MIN_GAP + TIME_HEADWAY * v_follower
        accel_command = K_GAP * (gap_perceived - desired_gap)
                      + K_SPEED * (v_leader_perceived - v_follower)

    Crashes occur when the lag means the command arrives too late for the gap available. That is a
    physically explicit, interpretable mechanism — unlike a Traffic Manager follower, which never
    crashes because it has no perception lag at all.
    """

    def __init__(self, actor, reaction_ticks):
        self.actor = actor
        self.history = deque(maxlen=reaction_ticks + 1)   # oldest entry = delayed observation

    def alive(self):
        return self.actor is not None and self.actor.is_alive

    def observe(self, ego):
        """Record the ego's true state this tick. The follower will act on the OLDEST entry."""
        if not self.alive():
            return
        tf = ego.get_transform()
        self.history.append((tf.location.x, tf.location.y, speed_of(ego)))

    def current_gap(self, ego):
        """True (unlagged) longitudinal gap, for logging only — never used for control."""
        if not self.alive():
            return None
        tf = ego.get_transform()
        fwd = tf.get_forward_vector()
        d = self.actor.get_transform().location
        longi = (d.x - tf.location.x) * fwd.x + (d.y - tf.location.y) * fwd.y
        return round(max(0.0, abs(longi) - 4.5), 3)

    def gap_valid(self, ego):
        """AMENDMENT 9: is the longitudinal projection a meaningful following distance?

        current_gap() projects onto the ego's forward axis, which is only a following distance while
        the follower is actually behind the ego, roughly in line, and pointing the same way. Pilot 8
        logged four readings above 40 m — including one trace that ran 13.7 -> 67.6 -> 2.9 m, which
        is not a car-following trajectory but a follower that lost the ego round a corner and was
        later re-projected. Those readings were silently averaged into the position means. This flags
        them instead: lateral offset within a lane width, follower genuinely behind, heading aligned.
        """
        if not self.alive():
            return False
        tf = ego.get_transform()
        fwd = tf.get_forward_vector()
        d = self.actor.get_transform().location
        dx, dy = d.x - tf.location.x, d.y - tf.location.y
        longi = dx * fwd.x + dy * fwd.y
        lat = abs(-dx * fwd.y + dy * fwd.x)
        dyaw = abs((self.actor.get_transform().rotation.yaw - tf.rotation.yaw + 180) % 360 - 180)
        return bool(longi < 0 and abs(longi) < 40.0 and lat < 3.0 and dyaw < 35.0)

    def step(self, world, ego):
        """One tick of control.

        AMENDMENT 4 — lag applies to the lead vehicle's SPEED, not its POSITION.

        The previous version lagged position, which had a fatal side effect: perceiving the ego
        ~v*lag metres closer than it really is makes the controller settle that much FURTHER BACK.
        Measured: desired 7.5 m + 9.3 m lag travel = 16.8 m predicted equilibrium, 20.7 m observed.
        So the delay that was meant to CAUSE collisions was pushing the follower out of collision
        range by the same amount — the two effects cancel, and no reaction-delay value escapes it.
        That is why three separate parameter fixes all produced zero crashes.

        Lagging speed instead removes the steady-state bias (gap is measured truthfully, so the
        follower genuinely tailgates at the desired distance) while preserving the dynamic delay
        (it does not notice the ego decelerating for 1.25 s). It is also the better driver model:
        humans judge distance roughly correctly but are slow to register deceleration.
        """
        if not self.alive() or len(self.history) == 0:
            return
        lx, ly, lv = self.history[0]        # ego as observed reaction_ticks ago (pos AND speed)
        tf = self.actor.get_transform()
        fwd = tf.get_forward_vector()
        lagged_gap = ((lx - tf.location.x) * fwd.x + (ly - tf.location.y) * fwd.y) - 4.5

        # AMENDMENT 5 — CONSTANT-VELOCITY EXTRAPOLATION FROM A DELAYED OBSERVATION.
        #
        # Amendments 1-4 each failed for a reason that is now fully understood:
        #   lag the POSITION      -> follower settles v*lag further back  -> out of range, no crash
        #   lag only the SPEED    -> gap term sees closure instantly      -> brakes in time, no crash
        # Both are structural, not parametric. What is required is an UNBIASED steady-state gap AND a
        # DELAYED response to closure, simultaneously.
        #
        # This is the standard delayed-perception driver model: the follower sees where the ego was
        # tau seconds ago and assumes it MAINTAINED that speed since.
        #   steady state: lagged_gap = true_gap - v*tau, so the correction cancels exactly and the
        #                 perceived gap equals the true gap -> follower tailgates at the desired
        #                 distance with no bias.
        #   hard braking: the ego actually decelerated, but the follower extrapolates its old, higher
        #                 speed forward, so it OVERESTIMATES the remaining gap and fails to brake.
        #                 Worked example, 8 m/s -> 0 stop: true gap collapses to ~1.9 m while the
        #                 perceived gap still reads ~7.9 m. That is the collision.
        #   light braking: ego barely slows, extrapolation error is small, no collision. This is the
        #                 severity-dependent variation the hazard measure needs.
        lag_s = (len(self.history) - 1) * 0.05      # actual lag available (buffer may still be filling)
        gap = lagged_gap + lv * lag_s
        vf = speed_of(self.actor)

        desired = FOLLOWER_MIN_GAP + FOLLOWER_TIME_HEADWAY * vf
        accel = FOLLOWER_K_GAP * (gap - desired) + FOLLOWER_K_SPEED * (lv - vf)

        # free-flow cap: don't accelerate past target speed when unconstrained
        if vf > FOLLOWER_TARGET_SPEED and accel > 0:
            accel = 0.0

        # AMENDMENT 3: was min(0.7, accel/3.0) — too weak to close a gap against a Model 3 even at
        # full command. Allow full throttle and a steeper mapping so the controller has the
        # authority to actually hold station.
        if accel >= 0:
            throttle, brake = min(1.0, accel / 2.0), 0.0
        else:
            throttle, brake = 0.0, min(1.0, -accel / 5.0)

        apply_control(self.actor, throttle, brake, lane_follow_steer(self.actor, world))


def spawn_follower(world, blueprint_library, ego, gap, rng):
    """One follower per EPISODE, behind the ego, matched to traffic speed at spawn.
    NOT under Traffic Manager — see AMENDMENT 1."""
    tf = ego.get_transform()
    wp = world.get_map().get_waypoint(tf.location, project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
    if wp is None:
        return None
    back = wp.previous(gap)
    if not back:
        return None
    b = back[0].transform
    spawn_tf = carla.Transform(
        carla.Location(x=b.location.x, y=b.location.y, z=b.location.z + 0.3), b.rotation)

    for actor in world.get_actors().filter('vehicle.*'):
        if actor.get_transform().location.distance(spawn_tf.location) < 5.0:
            return None

    cands = blueprint_library.filter(FOLLOWER_BLUEPRINT)
    if not cands:
        return None
    npc = world.try_spawn_actor(cands[0], spawn_tf)
    if npc is None:
        return None

    npc.set_autopilot(False)
    fwd = npc.get_transform().get_forward_vector()
    npc.set_target_velocity(carla.Vector3D(x=fwd.x * FOLLOWER_TARGET_SPEED,
                                           y=fwd.y * FOLLOWER_TARGET_SPEED, z=0.0))
    return Follower(npc, FOLLOWER_REACTION_TICKS)


# ── Phases ───────────────────────────────────────────────────────────────────

def phase(world, ego, spectator, fol, n_ticks, throttle, brake):
    """Run n_ticks with fixed ego throttle/brake plus lane-following steer, stepping the follower
    each tick. The EGO is traffic-blind by design: the decision must not react to the follower, or
    the treatment stops being a clean manipulation."""
    for _ in range(n_ticks):
        apply_control(ego, throttle, brake, lane_follow_steer(ego, world))
        if fol is not None:
            fol.observe(ego)
            fol.step(world, ego)
        world.tick()
        move_spectator(world, ego, spectator)


def spin_up(world, ego, spectator, fol):
    """AMENDMENT 9: accelerate until the ego regains ENTRY_SPEED_TARGET, capped.

    Returns (ticks_used, entry_speed). Without this the ego never recovers from a stop-hold and the
    whole chain decays to walking pace by t=3, which is what made late-position crashes impossible
    in pilot 8. The follower is stepped throughout and is NOT reset — only the ego's speed is
    equalised, so any carry-forward in the follower's state survives into the next decision.
    """
    used = 0
    while speed_of(ego) < ENTRY_SPEED_TARGET and used < MAX_SPINUP_TICKS:
        apply_control(ego, 1.0, 0.0, lane_follow_steer(ego, world))
        if fol is not None:
            fol.observe(ego)
            fol.step(world, ego)
        world.tick()
        move_spectator(world, ego, spectator)
        used += 1
    return used, round(speed_of(ego), 3)


def do_decision(world, ego, spectator, fol, sev_rng, writer,
                episode_id, arm, position, collision_state, hold_stats=None):
    """One brake decision: spin-up -> cruise -> brake (sampled severity) -> sample -> recover."""
    spinup_ticks, _ = spin_up(world, ego, spectator, fol)
    phase(world, ego, spectator, fol, TICKS_CRUISE, CRUISE_THROTTLE, 0.0)
    entry_speed = round(speed_of(ego), 3)

    intensity = round(sev_rng.uniform(*BRAKE_INTENSITY_RANGE), 3)
    ticks     = sev_rng.randint(*BRAKE_DURATION_RANGE)
    severity  = round(intensity * ticks, 4)

    collision_state['hit'] = False
    collision_state['actor'] = None
    gap_before  = fol.current_gap(ego) if fol else None
    gap_valid   = fol.gap_valid(ego) if fol else False
    fol_v_before = round(speed_of(fol.actor), 3) if (fol and fol.alive()) else None

    phase(world, ego, spectator, fol, ticks, 0.0, intensity)

    # AMENDMENT 6: if the brake actually brought the ego near a halt, HOLD the stop. This converts a
    # brake pulse into an emergency stop, which is the situation that actually produces rear-end
    # collisions. Severity-dependent by construction — light brakes never reach STOP_HOLD_SPEED and
    # so get no dwell, which preserves the variation the hazard measure depends on.
    held = 0
    if speed_of(ego) < STOP_HOLD_SPEED:
        phase(world, ego, spectator, fol, TICKS_STOP_HOLD, 0.0, 1.0)
        held = TICKS_STOP_HOLD

    # AMENDMENT 8: record the REALISED halt rate so ISO-LATE's time-matching tracks it instead of a
    # hardcoded guess. The old 0.15 constant was read off pilot 6 and went stale the moment the brake
    # duration range changed — exactly the kind of frozen assumption that produced this project's
    # earlier failures. Only CHAIN decisions count, because CHAIN position 7 is what ISO-LATE matches.
    # AMENDMENT 9: accumulate the REALISED cycle length too. Spin-up is variable, so ISO-LATE can no
    # longer be time-matched from constants at all — it is matched to the measured CHAIN cycle.
    if hold_stats is not None and arm == 'CHAIN':
        hold_stats['n'] += 1
        hold_stats['held'] += 1 if held else 0
        hold_stats['ticks'] += spinup_ticks + TICKS_CRUISE + ticks + held + TICKS_RECOVER

    s_t = get_state(ego, world)

    phase(world, ego, spectator, fol, TICKS_RECOVER, CRUISE_THROTTLE, 0.0)
    s_t1 = get_state(ego, world)

    crashed = collision_state['hit']
    writer.writerow({
        'episode_id': episode_id, 'arm': arm, 'position': position,
        'setting_cruise_ticks': TICKS_CRUISE,
        'brake_intensity': intensity, 'brake_ticks': ticks, 'severity': severity,
        'entry_speed': entry_speed, 'spinup_ticks': spinup_ticks,
        'follower_speed_before': fol_v_before, 'follower_gap_valid': gap_valid,
        'speed_t': s_t['speed'], 'speed_t1': s_t1['speed'],
        'x_t1': s_t1['x'], 'y_t1': s_t1['y'],
        'road_id_t1': s_t1['road_id'], 'is_junction_t1': s_t1['is_junction'],
        'follower_present': bool(fol and fol.alive()),
        'follower_gap_before': gap_before,
        'follower_gap_after': fol.current_gap(ego) if fol else None,
        'follower_speed': round(speed_of(fol.actor), 3) if (fol and fol.alive()) else None,
        'stop_hold_ticks': held,
        'outcome': 'crash' if crashed else 'safe',
        'hit_actor': collision_state.get('actor'),
    })
    return crashed


def run_chain(world, ego, spectator, fol, sev_rng, writer, episode_id, collision_state,
              hold_stats=None):
    """CHAIN arm: up to 8 decisions. Position 0 doubles as the ISO-EARLY baseline."""
    for pos in range(DECISIONS_PER_CHAIN):
        if do_decision(world, ego, spectator, fol, sev_rng, writer,
                       episode_id, 'CHAIN', pos, collision_state, hold_stats):
            return pos + 1, True
    return DECISIONS_PER_CHAIN, False


def run_iso_late(world, ego, spectator, fol, sev_rng, writer, episode_id, collision_state,
                 hold_stats=None):
    """ISO-LATE arm: cruise for a duration matched to 7 CHAIN cycles, then ONE decision.

    Timing is matched by drawing 7 brake durations from the SAME distribution and cruising the
    equivalent total, so elapsed time is distributionally matched to CHAIN position 7 rather than
    matched only at the mean. These draws are consumed for timing and are not decisions."""
    # AMENDMENT 6: include the expected stop-hold dwell so ISO-LATE stays time-matched to
    # CHAIN position 7.
    # AMENDMENT 8: the dwell weighting is now the halt rate OBSERVED in this run's CHAIN arms rather
    # than a constant. 0.26 is the prior used until 20 CHAIN decisions have been seen; it is the rate
    # the widened brake-duration range is expected to produce, and it is replaced by measurement as
    # soon as measurement exists.
    # AMENDMENT 9: matched to the MEASURED mean CHAIN cycle, because spin-up is variable and no
    # closed-form expression for cycle length exists any more. Falls back to the constant-based
    # estimate only until 20 CHAIN decisions have been observed.
    if hold_stats and hold_stats.get('n', 0) >= 20:
        cycle = hold_stats['ticks'] / hold_stats['n']
    else:
        cycle = (TICKS_CRUISE + statistics.mean(BRAKE_DURATION_RANGE) + TICKS_RECOVER
                 + 0.26 * TICKS_STOP_HOLD + 40)
    matched = int(cycle * (DECISIONS_PER_CHAIN - 1))
    while matched > 0:
        step = min(40, matched)
        phase(world, ego, spectator, fol, step, CRUISE_THROTTLE, 0.0)
        matched -= step
    crashed = do_decision(world, ego, spectator, fol, sev_rng, writer,
                          episode_id, 'ISO_LATE', DECISIONS_PER_CHAIN - 1, collision_state)
    return 1, crashed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global TICKS_CRUISE          # AMENDMENT 10: rebound per sweep setting
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sev_rng   = random.Random(SEV_RNG_SEED)
    spawn_rng = random.Random(SPAWN_RNG_SEED)
    # AMENDMENT 8: realised stop-hold rate, accumulated over CHAIN decisions and used by ISO-LATE
    # for time-matching. Also printed at the end as a direct check on the brake-range widening.
    hold_stats = {'n': 0, 'held': 0, 'ticks': 0}

    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT)
    world = client.get_world()
    print(f"Connected. Map: {world.get_map().name}")
    print(f"PILOT_MODE={PILOT_MODE}  episodes/arm={EPISODES_PER_ARM}")
    print(f"follower: reaction {FOLLOWER_REACTION_TICKS} ticks "
          f"({FOLLOWER_REACTION_TICKS*0.05:.2f} s), spawn gap "
          f"{FOLLOWER_GAP_RANGE[0]:.0f}-{FOLLOWER_GAP_RANGE[1]:.0f} m")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(TM_SEED)

    bl = world.get_blueprint_library()
    ego_bp = bl.filter('vehicle.tesla.model3')[0]
    ego_bp.set_attribute('color', '255,50,50')
    collision_bp = bl.find('sensor.other.collision')
    spawn_points = world.get_map().get_spawn_points()
    spectator = world.get_spectator()

    fields = ['episode_id', 'arm', 'setting_cruise_ticks', 'position',
              'brake_intensity', 'brake_ticks', 'severity',
              'entry_speed', 'spinup_ticks', 'follower_speed_before',
              'speed_t', 'speed_t1', 'x_t1', 'y_t1', 'road_id_t1', 'is_junction_t1',
              'follower_present', 'follower_gap_before', 'follower_gap_after', 'follower_speed',
              'follower_gap_valid', 'stop_hold_ticks',
              'outcome', 'hit_actor']
    stats_fields = ['episode_id', 'arm', 'spawn_x', 'spawn_y', 'follower_spawned',
                    'decisions_run', 'crashed', 'status']
    failures = []

    try:
        with open(OUTPUT_FILE, 'w', newline='') as cf, open(EPISODE_STATS_FILE, 'w', newline='') as sf:
            w  = csv.DictWriter(cf, fieldnames=fields);        w.writeheader()
            sw = csv.DictWriter(sf, fieldnames=stats_fields);  sw.writeheader()

            for setting in SWEEP_CRUISE_TICKS:
                # AMENDMENT 10: TICKS_CRUISE is the inter-decision interval lever. Rebinding the module
                # global is deliberate — do_decision() reads it at call time, so the whole pipeline
                # follows without threading a parameter through five call sites.
                TICKS_CRUISE = setting
                hold_stats['n'] = hold_stats['held'] = hold_stats['ticks'] = 0   # cycle length changed
                interval_s = (8 + setting + TICKS_RECOVER) * 0.05
                print(f"\n{'#'*70}\n# SWEEP SETTING: TICKS_CRUISE={setting}  "
                      f"-> inter-decision interval ~{interval_s:.2f} s "
                      f"({interval_s/TAU_ESTIMATE_S:.2f} tau)   "
                      f"predicted persistence {math.exp(-interval_s/TAU_ESTIMATE_S):.2f}\n{'#'*70}",
                      flush=True)

                for i in range(EPISODES_PER_ARM * 2):
                    arm = 'CHAIN' if i % 2 == 0 else 'ISO_LATE'
                    ep  = setting * 1000 + i + 1
                    spawn_point = spawn_points[i % len(spawn_points)]
                    ego = sensor = None
                    fol = None
                    collision_state = {'hit': False, 'actor': None}

                    def on_collision(event):
                        collision_state['hit'] = True
                        try:
                            collision_state['actor'] = event.other_actor.type_id
                        except Exception:
                            collision_state['actor'] = 'unknown'

                    try:
                        ego = world.try_spawn_actor(ego_bp, spawn_point)
                        if ego is None:
                            raise RuntimeError('ego spawn blocked')
                        sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)
                        sensor.listen(on_collision)

                        print(f"Episode {ep}/{EPISODES_PER_ARM*2} [{arm}] "
                              f"spawn ({spawn_point.location.x:.1f},{spawn_point.location.y:.1f})",
                              flush=True)

                        phase(world, ego, spectator, None, WARMUP_TICKS, CRUISE_THROTTLE, 0.0)

                        gap = spawn_rng.uniform(*FOLLOWER_GAP_RANGE)
                        fol = spawn_follower(world, bl, ego, gap, spawn_rng)
                        for _ in range(10):
                            if fol:
                                fol.observe(ego); fol.step(world, ego)
                            world.tick()

                        if arm == 'CHAIN':
                            n, crashed = run_chain(world, ego, spectator, fol, sev_rng,
                                                   w, ep, collision_state, hold_stats)
                        else:
                            n, crashed = run_iso_late(world, ego, spectator, fol, sev_rng,
                                                      w, ep, collision_state, hold_stats)

                        sw.writerow({'episode_id': ep, 'arm': arm,
                                     'spawn_x': round(spawn_point.location.x, 2),
                                     'spawn_y': round(spawn_point.location.y, 2),
                                     'follower_spawned': fol is not None,
                                     'decisions_run': n, 'crashed': crashed, 'status': 'ok'})
                        print(f"  decisions={n} crashed={crashed} follower={fol is not None}",
                              flush=True)

                    except Exception as e:
                        msg = f"Episode {ep} [{arm}] FAILED: {type(e).__name__}: {e}"
                        print('  ' + msg, flush=True)
                        log_error(msg)
                        failures.append(ep)
                        sw.writerow({'episode_id': ep, 'arm': arm,
                                     'spawn_x': round(spawn_point.location.x, 2),
                                     'spawn_y': round(spawn_point.location.y, 2),
                                     'follower_spawned': fol is not None,
                                     'decisions_run': None, 'crashed': None, 'status': 'exception'})
                    finally:
                        if sensor is not None:
                            sensor.stop(); sensor.destroy()
                        for a in ([fol.actor] if fol else []) + ([ego] if ego else []):
                            try:
                                a.destroy()
                            except Exception:
                                pass
                        for _ in range(10):
                            world.tick()
                        for f in (cf, sf):
                            f.flush(); os.fsync(f.fileno())

        # ── Pilot gates ──────────────────────────────────────────────────────
        with open(OUTPUT_FILE) as f:
            rows = list(csv.DictReader(f))
        n = len(rows)
        crashes = sum(1 for r in rows if r['outcome'] == 'crash')
        chain = [r for r in rows if r['arm'] == 'CHAIN']
        iso   = [r for r in rows if r['arm'] == 'ISO_LATE']

        # ── GATE 0 — printed FIRST because it blocked pilots 1, 2 and 3 ──────
        # If the follower is not in position, crash rate is meaningless and no reaction-delay
        # setting can matter. Read this before anything else.
        print(f"\n{'='*70}\nGATE 0 (READ FIRST) — IS THE FOLLOWER ACTUALLY IN POSITION?\n{'='*70}")
        gap_by_pos = {}
        for r in chain:
            g = r['follower_gap_before']
            if g not in ('', 'None', None):
                gap_by_pos.setdefault(int(r['position']), []).append(float(g))
        for p in sorted(gap_by_pos):
            v = gap_by_pos[p]
            print(f"   t={p}: gap mean {statistics.mean(v):6.1f} m   min {min(v):5.1f}")
        spd = [float(r['follower_speed']) for r in rows
               if r['follower_speed'] not in ('', 'None', None)]
        if spd:
            print(f"   follower speed: mean {statistics.mean(spd):.2f}  "
                  f"min {min(spd):.2f}  max {max(spd):.2f} m/s")
        # ── GATE 0, RECALIBRATED (AMENDMENT 8) ───────────────────────────────
        # The previous threshold was "mean gap under 15 m", a number chosen by intuition rather than
        # derived from anything. It passed in every pilot that produced zero crashes, which makes it
        # a gate that cannot fail for the right reason — the same defect as the Scenario D "control"
        # in the closed project. The physically correct threshold is the distance the follower can
        # actually close during a decision: a collision is possible only when the standing gap is
        # smaller than the achievable closure. Both quantities are now measured from this run.
        allg = [x for v in gap_by_pos.values() for x in v]
        closures = []
        for r in rows:
            gb, ga = r.get('follower_gap_before'), r.get('follower_gap_after')
            if gb not in ('', 'None', None) and ga not in ('', 'None', None):
                closures.append(float(gb) - float(ga))
        if allg and closures:
            closures.sort()
            c_p90 = closures[int(0.9 * (len(closures) - 1))]
            c_max = closures[-1]
            in_range = sum(1 for g in allg if g < c_p90) / len(allg)
            print(f"\n   GATE 0 — is a collision physically reachable?")
            print(f"   gap: mean {statistics.mean(allg):.1f} m  min {min(allg):.1f} m")
            print(f"   closure achieved per decision: mean {statistics.mean(closures):.2f} m  "
                  f"p90 {c_p90:.2f} m  max {c_max:.2f} m")
            print(f"   decisions starting inside p90 closure: {100*in_range:.1f}%")
            if in_range >= 0.25:
                print("   -> PASS: a meaningful share of decisions can actually reach contact.")
            else:
                print("   -> FAIL: the follower cannot close the standing gap. Any crash rate")
                print("      below is a floor effect, not a measurement. Fix geometry, not stats.")

        print(f"\n{'='*70}\nPILOT GATES\n{'='*70}")
        print(f"decisions logged: {n} | crashes: {crashes} ({100*crashes/max(n,1):.1f}%)")
        # AMENDMENT 9: upper bound narrowed 15% -> 9%. GATE 2 needs (1-p)^7 >= 0.5, i.e. p <= 9.4%,
        # so the original [5%, 15%] band was not jointly satisfiable with GATE 2 — pilot 8 failed
        # GATE 2 at p=10.9% purely by arithmetic. Writing two gates without checking they can both
        # hold is the same error class the gates exist to catch.
        print("  GATE 1 (PRIMARY): 5-9% per decision (upper bound set by GATE 2's (1-p)^7 >= 0.5).")
        held_n = sum(1 for r in rows if r.get('stop_hold_ticks') not in ('', '0', 0, None)
                     and int(r['stop_hold_ticks']) > 0)
        print(f"\n  halt rate (AMENDMENT 8 target ~26%): {held_n}/{n} = "
              f"{100*held_n/max(n,1):.1f}% of decisions brought the ego below "
              f"{STOP_HOLD_SPEED} m/s")
        if held_n:
            print(f"  crash-given-halt: {crashes}/{held_n} = "
                  f"{100*crashes/held_n:.1f}%  (pilot 6/7 measured ~30%)")
        print("  Read these two together: crash rate = halt rate x crash-given-halt. If GATE 1")
        print("  fails, this says which factor is responsible instead of leaving it to guesswork.")

        reach = {}
        for r in chain:
            p = int(r['position'])
            reach.setdefault(p, [0, 0])
            reach[p][0] += 1
            if r['outcome'] == 'crash':
                reach[p][1] += 1
        print("\nhazard by CHAIN position (crashes / reached):")
        for p in sorted(reach):
            tot, cr = reach[p]
            print(f"   t={p}: {cr:3d}/{tot:3d}  {100*cr/tot:5.1f}%")
        if iso:
            ic = sum(1 for r in iso if r['outcome'] == 'crash')
            print(f"\nISO_LATE: {ic}/{len(iso)} = {100*ic/len(iso):.1f}%   "
                  f"(primary comparison: CHAIN t=7 vs this)")

        n_chain_eps = len(set(r['episode_id'] for r in chain))
        if n_chain_eps:
            r7 = reach.get(7, [0, 0])[0]
            print(f"\nGATE 2: episodes reaching t=7: {r7}/{n_chain_eps} "
                  f"({100*r7/n_chain_eps:.0f}%) — need >=50%")

        print("\nGATE 3: severity must be FLAT across positions")
        for p in sorted(reach):
            sv = [float(r['severity']) for r in chain if int(r['position']) == p]
            if sv:
                print(f"   t={p}: mean severity {statistics.mean(sv):.2f}")

        # ── GATE 5 (AMENDMENT 9) — REGIME COMPARABILITY ──────────────────────
        # The gate pilot 8 lacked. Crashes need closing kinetic energy, so if ego or follower speed
        # decays across the chain, late positions sit in a regime where the outcome is near
        # impossible and the hazard curve reports the decay rather than the position effect.
        # CHAIN t=7 must be in the SAME physical regime as t=0 and as ISO-LATE, or the primary
        # comparison is meaningless regardless of what the crash counts say.
        print("\nGATE 5: regime comparability — ego entry speed and follower speed by position")
        print("   pos    n   entry_speed   follower_speed   valid_gap   mean gap(valid)")
        def col(rs, k):
            v = [float(r[k]) for r in rs if r.get(k) not in ('', 'None', None)]
            return statistics.mean(v) if v else float('nan')
        ent = {}
        for p in sorted(reach):
            sub = [r for r in chain if int(r['position']) == p]
            vg = [float(r['follower_gap_before']) for r in sub
                  if r.get('follower_gap_valid') == 'True'
                  and r['follower_gap_before'] not in ('', 'None', None)]
            ent[p] = col(sub, 'entry_speed')
            print("   t=%d  %3d   %11.2f   %14.2f   %4d/%-4d  %13s"
                  % (p, len(sub), ent[p], col(sub, 'follower_speed_before'), len(vg), len(sub),
                     ("%.1f" % statistics.mean(vg)) if vg else "n/a"))
        if iso:
            ivg = [float(r['follower_gap_before']) for r in iso
                   if r.get('follower_gap_valid') == 'True'
                   and r['follower_gap_before'] not in ('', 'None', None)]
            print("   ISO  %3d   %11.2f   %14.2f   %4d/%-4d  %13s"
                  % (len(iso), col(iso, 'entry_speed'), col(iso, 'follower_speed_before'),
                     len(ivg), len(iso), ("%.1f" % statistics.mean(ivg)) if ivg else "n/a"))
        if ent:
            spread = max(ent.values()) - min(ent.values())
            print("   entry-speed spread across positions: %.2f m/s "
                  "(need <= %.1f, else positions are not comparable)"
                  % (spread, ENTRY_SPEED_TOL))
            print("   -> %s" % ("PASS" if spread <= ENTRY_SPEED_TOL else
                                "FAIL: late decisions start slower; hazard curve is confounded "
                                "with speed decay, NOT a position effect"))

        # ── AMENDMENT 10 — THE DOSE-RESPONSE (this is the headline result) ───
        # Measures, per sweep setting, how much of a braking disturbance survives into the NEXT
        # decision, and compares it against the pre-registered exp(-interval/tau) prediction.
        # persistence = (gap_before[t] - gap_before[t+1]) / (gap_before[t] - gap_after[t])
        print(f"\n{'='*70}\nDOSE-RESPONSE: DISTURBANCE PERSISTENCE vs DECISION SPACING\n{'='*70}")
        print("  pre-registered model: persistence = exp(-interval / tau), tau = %.2f s\n"
              % TAU_ESTIMATE_S)
        print("  cruise  interval   /tau    n    applied  residual  PERSIST   PREDICTED   amp")
        by_set = {}
        for r in chain:
            by_set.setdefault(int(r['setting_cruise_ticks']), []).append(r)
        dose = []
        for st in sorted(by_set):
            ep_groups = {}
            for r in by_set[st]:
                ep_groups.setdefault(r['episode_id'], []).append(r)
            applied, residual, spin = [], [], []
            for _e, rs in ep_groups.items():
                rs.sort(key=lambda r: int(r['position']))
                for a, b in zip(rs, rs[1:]):
                    try:
                        gb, ga = float(a['follower_gap_before']), float(a['follower_gap_after'])
                        gn = float(b['follower_gap_before'])
                    except (ValueError, TypeError):
                        continue
                    if a.get('follower_gap_valid') != 'True':
                        continue
                    applied.append(gb - ga)
                    residual.append(gb - gn)
                for r in rs:
                    if r.get('spinup_ticks') not in ('', 'None', None):
                        spin.append(float(r['spinup_ticks']))
            if not applied or statistics.mean(applied) <= 0:
                continue
            mspin = statistics.mean(spin) if spin else 8.0
            iv = (mspin + st + TICKS_RECOVER) * 0.05
            pers = statistics.mean(residual) / statistics.mean(applied)
            pred = math.exp(-iv / TAU_ESTIMATE_S)
            amp = 1.0 / (1.0 - pers) if pers < 0.95 else float('inf')
            dose.append((st, iv, pers, pred, len(applied)))
            print("  %6d  %7.2fs  %5.2f  %4d  %7.2f  %8.2f  %7.3f  %9.3f  %5.2fx"
                  % (st, iv, iv / TAU_ESTIMATE_S, len(applied),
                     statistics.mean(applied), statistics.mean(residual), pers, pred, amp))
        if len(dose) >= 3:
            obs = [d[2] for d in dose]
            prd = [d[3] for d in dose]
            mo, mp = statistics.mean(obs), statistics.mean(prd)
            num = sum((a - mo) * (b - mp) for a, b in zip(obs, prd))
            da = math.sqrt(sum((a - mo) ** 2 for a in obs))
            db = math.sqrt(sum((b - mp) ** 2 for b in prd))
            print("\n  observed vs predicted persistence: r = %+.3f  (n=%d settings)"
                  % (num / (da * db) if da and db else 0.0, len(dose)))
            mono = all(obs[i] >= obs[i + 1] for i in range(len(obs) - 1))
            print("  monotone decreasing with spacing: %s" % ("YES" if mono else "NO"))
            print("\n  -> %s" % ("SUPPORTED: carry-forward is governed by a relaxation time "
                                 "constant,\n     not by decision count. Independent evaluation is "
                                 "valid above ~2 tau." if mono else
                                 "NOT SUPPORTED: persistence does not fall monotonically with "
                                 "spacing;\n     the exponential-relaxation model is wrong. Report "
                                 "as measured."))

        hits = [r['hit_actor'] for r in rows if r['outcome'] == 'crash']
        print(f"\nGATE 4: hit_actor populated on {sum(1 for h in hits if h)}/{len(hits)} crashes")
        for a, c in Counter(h for h in hits if h).most_common(5):
            print(f"   {a}: {c}")


        if failures:
            print(f"\n{len(failures)} episode(s) failed: {failures}")
        print(f"\nData: {OUTPUT_FILE}")

    finally:
        print("Restoring async mode...")
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)
        tm.set_synchronous_mode(False)
        print("Done.")


if __name__ == '__main__':
    main()
