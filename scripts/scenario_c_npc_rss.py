"""
Scenario C variant: emergency braking WITH NPC TRAFFIC + RSS-margin logging.

Purpose (2026-07-14, see CLAUDE.md "RSS baseline" decision entry): answer two
standing reviewer objections in one run —
  1. "Empty world": the banked suite has NPC_DENSITY=0, so risk formulas are
     proxies for dangers whose agents don't exist. This run adds Traffic
     Manager vehicles as ambient traffic.
  2. "Hand-crafted risk formulas": alongside the UNCHANGED Scenario C rho,
     this run logs an RSS-based longitudinal safe-distance margin to the
     nearest same-lane vehicle ahead and behind, at both T and T+1. RSS
     (Shalev-Shwartz et al., 2017) is memoryless BY CONSTRUCTION — it scores
     each instant's inter-vehicle geometry with no knowledge of past
     decisions. If decision-T brake severity predicts the RSS margin at T+1
     (same episode-level ablation machinery as Table I), carry-forward is
     demonstrated on an industry-standard, externally-defined measure, not
     just our own formula.

IMPORTANT CONSTRAINTS (do not violate):
  - CARLA's built-in RssSensor is LINUX-ONLY; the Windows 0.9.15 build ships
    without it. The RSS longitudinal formula is therefore implemented
    manually below (rss_d_min). In the paper call this an "RSS-based
    safe-distance margin", not "the CARLA RSS sensor".
  - Approach B is preserved: brake events fire ON SCHEDULE with sampled
    severity, regardless of traffic. NPCs are ambient disturbance, never
    decision triggers. The ONLY traffic-aware control is a headway guard
    during the cruise phase — cruise is transit, not a decision (same
    argument that justified autopilot navigation in Scenario B).
  - rho_actual math is byte-identical to the banked scenario_c_braking.py —
    RSS columns are ADDITIONS. Never merge this file with the banked one.

Output: scenario_c_npc_rss[_pilot].csv + episode-stats sidecar. The banked
scenario_c_braking.csv cannot be overwritten by this script.

REVISION (2026-07-21) — two fixes to the first pilot, found by inspecting
its own console gates + a code review, not by re-running blind:
  1. HEADING BUG: nearest_same_lane_neighbors() only filtered candidates by
     lateral offset + longitudinal distance, never checked heading. On a
     two-way road where the opposing lane's centerline sits inside
     SAME_LANE_LATERAL, an oncoming vehicle could get classified as a
     same-direction "front"/"rear" neighbor right up until it passed —
     which would corrupt the RSS margin (RSS's braking formula assumes a
     follower/leader in the SAME direction, not a closing head-on
     encounter). Fixed with a heading filter (HEADING_MAX_DIFF) and a
     rejection counter printed at run end so the fix's effect is visible,
     not assumed.
  2. SPARSE COVERAGE: the first pilot's RSS coverage was thin (19% front /
     32% rear present at T+1). The likely primary cause isn't N_NPC being
     too low in absolute terms — it's that NPCs were drawn uniformly from
     spawn_points[20:], i.e. anywhere on the whole map, so most of the 30
     NPCs never came near wherever the ego's episode actually happened to
     be. Fixed by biasing NPC spawn selection toward points near the
     ego's own spawn pool (NPC_SPAWN_RADIUS), with the full map as a
     fallback only if that pool can't fill N_NPC. Near/far spawn counts
     are printed so this is also verifiable, not assumed.
  Neither fix touches rho_actual, the brake-decision logic, or
  Approach B — both are confined to the neighbor-detection and NPC-spawn
  code paths.

REVISION 2 (2026-07-21, same day) — second pilot ran clean (0 failures,
heading rejection rate 11.0% over 3219 checks, confirming fix #1 above was
real) but surfaced a WORSE problem than "coverage is thin": coverage is
CONCENTRATED, not just sparse. Per-episode breakdown of the second pilot's
own CSV: 8/20 episodes had ZERO RSS-covered pairs at all (front and rear
both empty, all 8 decision pairs), and a single episode (ep 10) supplied
8 of the run's 17 total rear hits — 47% of all "rear coverage" in the
entire pilot came from one episode. rear_present dropped 32%->11% and
front barely moved (19%->21%) after fix #1, which fits: a lot of the old
"rear" hits were oncoming traffic caught mid-pass (now correctly excluded
by the heading filter), not genuine same-lane followers, which were
always rare. Root cause: spawn_npcs() places NPCs near the ego's spawn
POOL once, at the start of the whole ~20-episode run, then Traffic
Manager drives them off on independent routes for the full ~9-minute run
while the ego teleports between 20 different spawn points every ~27s.
"Near at spawn" says nothing about "near during episode 14." NPC_SPAWN_RADIUS
(fix #2 above) was treating a run-duration problem as a spawn-location
problem — necessary but not sufficient.
  Fix: ESCORT NPCs, spawned FRESH each episode (not once for the whole
  run) directly on the same road_id + lane_id as THAT episode's specific
  spawn point (escort_spawn_candidates(), matched via the map's own
  waypoint data, not Euclidean distance — proximity alone doesn't imply
  same lane, e.g. the opposite-direction lane of a two-way road can sit a
  few meters away). Escorts are destroyed at episode end alongside the
  ego, same lifecycle. The existing global ambient pool (N_NPC=30,
  NPC_SPAWN_RADIUS-biased) is kept unchanged for general background
  traffic / the empty-world objection; escorts are additive, purely to
  guarantee each episode starts with a real chance of same-lane coverage
  instead of depending on whatever the ambient pool has wandered into by
  that point in the run.
  Also added: Gate 4, per-episode coverage distribution (fraction of
  episodes with zero coverage, and the busiest single episode's share of
  total hits) — the aggregate presence % gates (Gate 2) cannot detect
  concentration like ep 10's, only a per-episode breakdown can. This is
  now computed and printed automatically instead of requiring a manual
  CSV pull after the fact.
  Still PILOT_MODE=True: re-pilot and check Gate 4 specifically before
  deciding anything else needs tuning. If zero-coverage episodes are
  still common with escorts in place, the next suspect is
  ESCORT_RADIUS/ESCORT_PER_EPISODE or SAME_LANE_LATERAL being too tight,
  not N_NPC.
"""
import carla
import math
import random
import csv
import os
import datetime
import statistics

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOST                  = 'localhost'
PORT                  = 2000
TIMEOUT               = 30.0

PILOT_MODE            = False
NUM_EPISODES          = 20 if PILOT_MODE else 100
# ──────────────────────────────────────────────────────────────────────────────

DECISIONS_PER_EPISODE = 8
TICKS_CRUISE          = 40
TICKS_RECOVER         = 10     # unchanged — T+1 sampled mid-recovery
WARMUP_TICKS          = 50
CRUISE_THROTTLE       = 0.5

# Severity sampling — identical to the banked run (metric locked a priori:
# intensity * duration; see scenario_c_braking.py for the full history).
BRAKE_INTENSITY_RANGE = (0.15, 0.9)
BRAKE_DURATION_RANGE  = (5, 20)

# ── NPC traffic ──────────────────────────────────────────────────────────────
N_NPC                 = 30     # ambient Traffic Manager vehicles (pilot-tunable:
                               # the pilot's NPC-coverage gate decides if this
                               # is enough for RSS margins to exist often)
TM_PORT               = 8000
TM_SEED               = 42
NPC_SPAWN_OFFSET      = 20     # spawn_points[:20] reserved for the ego (same
                               # list the banked run used); NPCs use the rest
NPC_SPAWN_RADIUS      = 100.0  # m — prefer NPC spawn points within this radius
                               # of an ego spawn point over the rest of the map.
                               # ~100-140m is roughly one episode's total travel
                               # distance (warmup + 8 cruise/brake/recover
                               # cycles at ego's typical ~4-5 m/s), so this
                               # targets where the ego will actually be, not
                               # just where a spawn point happens to exist.
                               # Full map is still the fallback if the near
                               # pool can't fill N_NPC.
                               # NOTE: this pool is spawned ONCE for the whole
                               # run and then wanders under TM — see ESCORT_*
                               # below for the per-episode fix that actually
                               # addresses coverage concentration.

# Escort NPCs — spawned fresh each episode on the SAME road_id + lane_id as
# that episode's own spawn point (see REVISION 2, module docstring). This is
# what actually guarantees a same-lane neighbor can exist near a given
# episode's decisions, rather than hoping the ambient pool above happens to
# still be nearby by the time that episode runs.
ESCORT_PER_EPISODE    = 4      # NPCs spawned/destroyed per episode, in addition
                               # to the ambient pool
ESCORT_RADIUS         = 40.0   # m — tight radius around THIS episode's spawn
                               # point; must also match road_id+lane_id exactly
                               # (see escort_spawn_candidates) — proximity alone
                               # doesn't imply same lane (e.g. the opposite
                               # lane of a two-way road can be a few meters
                               # away and still fail this filter, correctly)

# Headway guard (cruise/warmup phases ONLY — see module docstring):
HEADWAY_BRAKE_GAP     = 12.0   # m — proportional brake when front gap below this
HEADWAY_FULL_GAP      = 5.0    # m — full brake at/below this

# ── RSS parameters (Shalev-Shwartz et al. 2017, longitudinal case) ──────────
# d_min = v_r*rho + 0.5*a_max*rho^2 + (v_r + rho*a_max)^2/(2*b_min)
#         - v_f^2/(2*b_max),  clamped at >= 0
# where v_r = rear vehicle speed, v_f = front vehicle speed.
# margin = actual_gap - d_min. Negative margin = RSS violation.
RSS_RHO               = 1.0    # s   — response time
RSS_A_MAX_ACCEL       = 3.5    # m/s^2 — max acceleration during response
RSS_B_MIN_BRAKE       = 4.0    # m/s^2 — min (guaranteed) braking of rear car
RSS_B_MAX_BRAKE       = 8.0    # m/s^2 — max braking of front car
NEIGHBOR_RANGE        = 60.0   # m — how far to scan for same-lane vehicles
SAME_LANE_LATERAL     = 2.0    # m — lateral tolerance for "same lane"
VEHICLE_LENGTH_COMP   = 4.5    # m — bumper-to-bumper correction (two half-lengths)
HEADING_MAX_DIFF      = 30.0   # deg — reject a geometric candidate whose heading
                               # differs from the ego's by more than this. Without
                               # it, an oncoming vehicle on a two-way road can fall
                               # inside SAME_LANE_LATERAL and get misread as a
                               # same-direction front/rear neighbor. 30 deg gives
                               # room for the ego's own heading drift through a
                               # curve while still excluding oncoming (~180 deg)
                               # and cross traffic (~90 deg).

RNG_SEED              = 42

OUTPUT_DIR            = r'C:\Users\jayad\Desktop\RPI_AINN\data'
OUTPUT_FILE           = os.path.join(
    OUTPUT_DIR,
    'scenario_c_npc_rss_pilot.csv' if PILOT_MODE else 'scenario_c_npc_rss.csv'
)
SCENARIO_TYPE         = 'C_braking_npc_rss'
WEATHER               = 'clear'
ERROR_LOG_FILE        = OUTPUT_FILE.replace('.csv', '_errors.log')
EPISODE_STATS_FILE    = OUTPUT_FILE.replace('.csv', '_episode_stats.csv')
# ──────────────────────────────────────────────────────────────────────────────


def log_error(msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")


def soft_cap(x, scale):
    return 1.0 - math.exp(-x / scale)


def get_risk_score(vehicle, world):
    """UNCHANGED from banked scenario_c_braking.py — byte-identical math."""
    velocity = vehicle.get_velocity()
    speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    inverse_speed = max(0.0, 1.0 - speed_ms / 15.0)
    transform = vehicle.get_transform()
    waypoint  = world.get_map().get_waypoint(
        transform.location, project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        lane_deviation = 1.0
    else:
        wp_loc = waypoint.transform.location
        dist   = math.sqrt((transform.location.x - wp_loc.x)**2 +
                           (transform.location.y - wp_loc.y)**2)
        lane_deviation = soft_cap(dist, scale=3.0)
    if waypoint is not None:
        yaw_diff = abs(transform.rotation.yaw - waypoint.transform.rotation.yaw)
        yaw_diff = min(yaw_diff, 360 - yaw_diff)
        yaw_risk = soft_cap(yaw_diff, scale=30.0)
    else:
        yaw_risk = 1.0
    risk = 0.4 * inverse_speed + 0.4 * lane_deviation + 0.2 * yaw_risk
    return round(min(risk, 0.999999), 6)


# ── RSS machinery ────────────────────────────────────────────────────────────

def rss_d_min(v_rear, v_front):
    """RSS minimum safe longitudinal distance (m). Clamped at >= 0."""
    d = (v_rear * RSS_RHO
         + 0.5 * RSS_A_MAX_ACCEL * RSS_RHO ** 2
         + (v_rear + RSS_RHO * RSS_A_MAX_ACCEL) ** 2 / (2.0 * RSS_B_MIN_BRAKE)
         - v_front ** 2 / (2.0 * RSS_B_MAX_BRAKE))
    return max(0.0, d)


def _speed_of(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


# Diagnostic counters for the 2026-07-21 heading fix — module-level so every
# call site (headway guard + RSS snapshots) contributes to one run-end total.
# Printed and logged at the end of main() so the fix's effect is verifiable
# from the run's own output, not just asserted in a comment.
heading_stats = {'checked': 0, 'rejected': 0}


def _heading_diff_deg(yaw_a, yaw_b):
    diff = abs(yaw_a - yaw_b) % 360.0
    return min(diff, 360.0 - diff)


def nearest_same_lane_neighbors(world, ego):
    """
    Nearest vehicle ahead of and behind the ego in (approximately) its own
    lane, by projecting relative positions onto the ego's forward/right
    axes, THEN filtering by heading (fixed 2026-07-21 — see module
    docstring "REVISION" note: geometric-only filtering let oncoming
    traffic on two-way roads get misclassified as a same-direction
    front/rear neighbor). Otherwise deliberately geometric rather than
    waypoint-lane-id based: lane ids flip across road-segment boundaries,
    and RSS cares about physical gaps, not OpenDRIVE bookkeeping.
    Returns (front, rear), each None or dict(gap, speed).
    """
    ego_tf  = ego.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = ego_tf.rotation.yaw
    fwd     = ego_tf.get_forward_vector()
    right   = ego_tf.get_right_vector()
    front, rear = None, None
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.id == ego.id:
            continue
        actor_tf = actor.get_transform()
        loc = actor_tf.location
        rel_x, rel_y = loc.x - ego_loc.x, loc.y - ego_loc.y
        longi = rel_x * fwd.x + rel_y * fwd.y
        lat   = abs(rel_x * right.x + rel_y * right.y)
        if lat > SAME_LANE_LATERAL or abs(longi) > NEIGHBOR_RANGE or abs(longi) < 0.5:
            continue
        heading_stats['checked'] += 1
        if _heading_diff_deg(actor_tf.rotation.yaw, ego_yaw) > HEADING_MAX_DIFF:
            heading_stats['rejected'] += 1
            continue   # oncoming or cross traffic, not a same-direction neighbor
        gap = max(0.0, abs(longi) - VEHICLE_LENGTH_COMP)
        if longi > 0 and (front is None or gap < front['gap']):
            front = {'gap': gap, 'speed': _speed_of(actor)}
        elif longi < 0 and (rear is None or gap < rear['gap']):
            rear = {'gap': gap, 'speed': _speed_of(actor)}
    return front, rear


def rss_snapshot(world, ego):
    """
    RSS state at one instant, raw values logged (no baked-in normalization —
    analysis decides offline, per house rules on capped components).
      front: ego is the REAR vehicle  -> d_min(v_ego, v_front)
      rear:  NPC is the rear vehicle  -> d_min(v_npc, v_ego)
    """
    ego_speed = _speed_of(ego)
    front, rear = nearest_same_lane_neighbors(world, ego)
    snap = {
        'front_present': front is not None, 'front_gap': None,
        'front_v': None, 'front_margin': None,
        'rear_present': rear is not None, 'rear_gap': None,
        'rear_v': None, 'rear_margin': None,
    }
    if front is not None:
        dmin = rss_d_min(ego_speed, front['speed'])
        snap.update(front_gap=round(front['gap'], 3),
                    front_v=round(front['speed'], 3),
                    front_margin=round(front['gap'] - dmin, 3))
    if rear is not None:
        dmin = rss_d_min(rear['speed'], ego_speed)
        snap.update(rear_gap=round(rear['gap'], 3),
                    rear_v=round(rear['speed'], 3),
                    rear_margin=round(rear['gap'] - dmin, 3))
    return snap


# ── State / control (banked patterns) ────────────────────────────────────────

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


def apply_control(vehicle, throttle, brake, steer=0.0):
    control          = carla.VehicleControl()
    control.throttle = throttle
    control.brake    = brake
    control.steer    = steer
    vehicle.apply_control(control)


LANE_FOLLOW_LOOKAHEAD = 6.0


def lane_follow_steer(vehicle, world, lookahead=LANE_FOLLOW_LOOKAHEAD):
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
    target_loc  = ahead[0].transform.location
    vehicle_yaw = math.radians(transform.rotation.yaw)
    yaw_diff = math.atan2(target_loc.y - transform.location.y,
                          target_loc.x - transform.location.x) - vehicle_yaw
    while yaw_diff >  math.pi: yaw_diff -= 2 * math.pi
    while yaw_diff < -math.pi: yaw_diff += 2 * math.pi
    return max(-1.0, min(1.0, yaw_diff / math.pi * 2.0))


def move_spectator(world, vehicle, spectator):
    loc   = vehicle.get_transform().location
    yaw   = vehicle.get_transform().rotation.yaw
    fwd_x = math.cos(math.radians(yaw))
    fwd_y = math.sin(math.radians(yaw))
    spectator.set_transform(carla.Transform(
        carla.Location(x=loc.x - 10 * fwd_x, y=loc.y - 10 * fwd_y, z=loc.z + 5),
        carla.Rotation(pitch=-20, yaw=yaw)
    ))


def cruise_with_headway(world, vehicle, spectator, n_ticks):
    """
    Cruise phase WITH a headway guard: proportional braking when the front
    gap closes below HEADWAY_BRAKE_GAP. This is the only traffic-aware
    control in the script, and it is deliberately confined to cruise/warmup
    (transit) — the brake decision itself remains scripted and
    traffic-blind, preserving Approach B. Without this, a fixed-throttle
    ego rear-ends slow NPCs before ever reaching a brake event.
    """
    for _ in range(n_ticks):
        steer = lane_follow_steer(vehicle, world)
        front, _ = nearest_same_lane_neighbors(world, vehicle)
        if front is not None and front['gap'] < HEADWAY_BRAKE_GAP:
            if front['gap'] <= HEADWAY_FULL_GAP:
                apply_control(vehicle, 0.0, 1.0, steer)
            else:
                frac = (HEADWAY_BRAKE_GAP - front['gap']) / (HEADWAY_BRAKE_GAP - HEADWAY_FULL_GAP)
                apply_control(vehicle, 0.0, min(1.0, frac), steer)
        else:
            apply_control(vehicle, CRUISE_THROTTLE, 0.0, steer)
        world.tick()
        move_spectator(world, vehicle, spectator)


def scripted_phase(world, vehicle, spectator, n_ticks, throttle, brake):
    """Brake / recover phases: fixed control + lane-follow steer, traffic-blind
    on purpose (these ARE the decisions)."""
    for _ in range(n_ticks):
        steer = lane_follow_steer(vehicle, world)
        apply_control(vehicle, throttle, brake, steer)
        world.tick()
        move_spectator(world, vehicle, spectator)


def run_episode(episode_id, vehicle, world, spectator, writer, rng, collision_state):
    cruise_with_headway(world, vehicle, spectator, WARMUP_TICKS)
    rows = []
    speed_t_values = []
    for d in range(DECISIONS_PER_EPISODE):
        cruise_with_headway(world, vehicle, spectator, TICKS_CRUISE)

        brake_intensity = round(rng.uniform(*BRAKE_INTENSITY_RANGE), 3)
        brake_ticks     = rng.randint(*BRAKE_DURATION_RANGE)
        collision_state['hit'] = False
        scripted_phase(world, vehicle, spectator, brake_ticks, 0.0, brake_intensity)

        s_t    = get_state(vehicle, world)
        rho_t  = get_risk_score(vehicle, world)
        rss_t  = rss_snapshot(world, vehicle)

        scripted_phase(world, vehicle, spectator, TICKS_RECOVER, CRUISE_THROTTLE, 0.0)

        s_t1   = get_state(vehicle, world)
        rho_t1 = get_risk_score(vehicle, world)
        rss_t1 = rss_snapshot(world, vehicle)

        transform = vehicle.get_transform()
        waypoint  = world.get_map().get_waypoint(
            transform.location, project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        off_road = True
        if waypoint is not None:
            wp_loc = waypoint.transform.location
            dist   = math.sqrt((transform.location.x - wp_loc.x) ** 2 +
                               (transform.location.y - wp_loc.y) ** 2)
            off_road = dist > 5.0
        outcome = 'crash' if collision_state['hit'] else ('off_road' if off_road else 'safe')

        row = {
            'episode_id': episode_id, 't': d,
            'x_t': s_t['x'], 'y_t': s_t['y'], 'speed_t': s_t['speed_ms'],
            'yaw_t': s_t['yaw'], 'lane_id_t': s_t['lane_id'], 'road_id_t': s_t['road_id'],
            'action_t': 'hard_brake',
            'brake_intensity_t': brake_intensity, 'brake_duration_ticks_t': brake_ticks,
            'rho_t': rho_t,
            'x_t1': s_t1['x'], 'y_t1': s_t1['y'], 'speed_t1': s_t1['speed_ms'],
            'yaw_t1': s_t1['yaw'], 'lane_id_t1': s_t1['lane_id'], 'road_id_t1': s_t1['road_id'],
            'rho_actual': rho_t1, 'rho_baseline': None, 'RPI_pair': None,
            'scenario': SCENARIO_TYPE, 'weather': WEATHER, 'npc_density': N_NPC,
            'outcome': outcome,
        }
        for prefix, snap in (('rss_t_', rss_t), ('rss_t1_', rss_t1)):
            for k, v in snap.items():
                row[prefix + k] = v
        writer.writerow(row)
        rows.append(row)
        speed_t_values.append(s_t['speed_ms'])

        if outcome == 'crash':
            break   # a crashed ego's later cycles are garbage; episode ends
    return rows, speed_t_values


def spawn_npcs(world, client, blueprint_library, spawn_points, ego_spawns):
    """
    Ambient TM traffic. 4-wheeled blueprints only (bikes fall over and
    block lanes). try_spawn so occupied points are skipped silently.

    Spawn points are biased toward the ego's own operating area (fixed
    2026-07-21 — see module docstring "REVISION" note: drawing uniformly
    from the whole map meant most of N_NPC=30 never came near wherever a
    given episode's ego actually was, which is the likely primary cause
    of the first pilot's thin RSS coverage). Candidates within
    NPC_SPAWN_RADIUS of ANY ego spawn point are tried first; the rest of
    the map is used only as a fallback if that pool can't fill N_NPC.
    Prints how many landed in each pool so the bias is verifiable.
    """
    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(TM_SEED)
    bps = [bp for bp in blueprint_library.filter('vehicle.*')
           if int(bp.get_attribute('number_of_wheels')) == 4]

    def is_near_ego(sp):
        return any(sp.location.distance(e.location) <= NPC_SPAWN_RADIUS
                   for e in ego_spawns)

    all_candidates = list(spawn_points[NPC_SPAWN_OFFSET:])
    # id()-based split, not `in`/`==` on Transform objects — CARLA transforms
    # don't guarantee value equality, only identity is safe to rely on here.
    near_flags = [is_near_ego(sp) for sp in all_candidates]
    near = [sp for sp, is_n in zip(all_candidates, near_flags) if is_n]
    far  = [sp for sp, is_n in zip(all_candidates, near_flags) if not is_n]
    rng  = random.Random(TM_SEED)
    rng.shuffle(near)
    rng.shuffle(far)
    ordered = [(sp, True) for sp in near] + [(sp, False) for sp in far]

    npcs = []
    near_used = 0
    for sp, is_near in ordered:
        if len(npcs) >= N_NPC:
            break
        npc = world.try_spawn_actor(rng.choice(bps), sp)
        if npc is not None:
            npc.set_autopilot(True, TM_PORT)
            npcs.append(npc)
            if is_near:
                near_used += 1
    print(f"  NPC spawn pool: {len(near)} near-ego candidates "
          f"(<= {NPC_SPAWN_RADIUS:.0f}m of an ego spawn), {len(far)} far. "
          f"Used {near_used}/{len(npcs)} from the near pool.")
    return tm, npcs


def escort_spawn_candidates(world, spawn_points, offset, ego_spawn_transform, radius):
    """
    Spawn points within `radius` of THIS episode's specific ego spawn point
    that also sit on the same road_id + lane_id (checked via the map's own
    waypoint data, not just Euclidean distance — a nearby point can easily
    be the opposite-direction lane of a two-way road, which is exactly the
    kind of candidate the heading fix now correctly excludes downstream, so
    there's no point spawning an escort there in the first place).
    """
    m = world.get_map()
    ego_wp = m.get_waypoint(ego_spawn_transform.location, project_to_road=True,
                             lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return []
    out = []
    for sp in spawn_points[offset:]:
        if sp.location.distance(ego_spawn_transform.location) > radius:
            continue
        wp = m.get_waypoint(sp.location, project_to_road=True,
                             lane_type=carla.LaneType.Driving)
        if wp is not None and wp.road_id == ego_wp.road_id and wp.lane_id == ego_wp.lane_id:
            out.append(sp)
    return out


def spawn_escorts(world, tm, blueprint_library, spawn_points, spawn_point, rng):
    """
    Per-episode escort NPCs — see ESCORT_* config and REVISION 2 in the
    module docstring. Returns a list of spawned actors (possibly fewer than
    ESCORT_PER_EPISODE, or empty, if this spawn point has no same-lane
    candidates within ESCORT_RADIUS — that's reported honestly, not forced).
    """
    bps = [bp for bp in blueprint_library.filter('vehicle.*')
           if int(bp.get_attribute('number_of_wheels')) == 4]
    candidates = escort_spawn_candidates(
        world, spawn_points, NPC_SPAWN_OFFSET, spawn_point, ESCORT_RADIUS)
    rng.shuffle(candidates)
    escorts = []
    for sp in candidates:
        if len(escorts) >= ESCORT_PER_EPISODE:
            break
        npc = world.try_spawn_actor(rng.choice(bps), sp)
        if npc is not None:
            npc.set_autopilot(True, TM_PORT)
            escorts.append(npc)
    return escorts


def destroy_escorts(escorts):
    for npc in escorts:
        try:
            npc.set_autopilot(False, TM_PORT)
            npc.destroy()
        except Exception:
            pass


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng        = random.Random(RNG_SEED)       # decision severity sampling ONLY
    escort_rng = random.Random(RNG_SEED + 1)   # escort candidate selection —
    # kept separate (2026-07-21) so escort spawning can't perturb the
    # severity RNG's draw sequence; severity sampling stays exactly isolated
    # from simulation-infrastructure randomness, same as the banked script.

    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT)
    world  = client.get_world()
    print(f"Connected. Map: {world.get_map().name}")
    print(f"PILOT_MODE = {PILOT_MODE}  (NUM_EPISODES = {NUM_EPISODES}, NPCs = {N_NPC})")

    settings                     = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp        = blueprint_library.filter('vehicle.tesla.model3')[0]
    vehicle_bp.set_attribute('color', '255,50,50')
    collision_bp      = blueprint_library.find('sensor.other.collision')
    spawn_points      = world.get_map().get_spawn_points()
    ego_spawns        = spawn_points[:NPC_SPAWN_OFFSET]   # same list as banked C
    spectator         = world.get_spectator()

    tm, npcs = spawn_npcs(world, client, blueprint_library, spawn_points, ego_spawns)
    print(f"Spawned {len(npcs)}/{N_NPC} NPC vehicles under Traffic Manager (sync)")
    for _ in range(40):   # let traffic disperse before episode 1
        world.tick()

    base_fields = [
        'episode_id', 't',
        'x_t', 'y_t', 'speed_t', 'yaw_t', 'lane_id_t', 'road_id_t',
        'action_t', 'brake_intensity_t', 'brake_duration_ticks_t', 'rho_t',
        'x_t1', 'y_t1', 'speed_t1', 'yaw_t1', 'lane_id_t1', 'road_id_t1',
        'rho_actual', 'rho_baseline', 'RPI_pair',
        'scenario', 'weather', 'npc_density', 'outcome',
    ]
    rss_fields = [p + k for p in ('rss_t_', 'rss_t1_')
                  for k in ('front_present', 'front_gap', 'front_v', 'front_margin',
                            'rear_present', 'rear_gap', 'rear_v', 'rear_margin')]
    fieldnames = base_fields + rss_fields
    stats_fields = ['episode_id', 'spawn_x', 'spawn_y', 'pairs_logged',
                    'crashes', 'transit_collisions', 'escorts_spawned', 'status']

    failed_episodes = []
    all_rows        = []
    all_speed_t     = []

    try:
        with open(OUTPUT_FILE, 'w', newline='') as csvfile, \
             open(EPISODE_STATS_FILE, 'w', newline='') as statsfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            stats_writer = csv.DictWriter(statsfile, fieldnames=stats_fields)
            stats_writer.writeheader()

            for ep in range(NUM_EPISODES):
                spawn_point      = ego_spawns[ep % len(ego_spawns)]
                vehicle          = None
                collision_sensor = None
                escorts          = []
                collision_state  = {'hit': False, 'total_hits': 0}

                def on_collision(event):
                    collision_state['hit'] = True
                    collision_state['total_hits'] += 1

                try:
                    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
                    if vehicle is None:
                        # an NPC is sitting on the ego spawn — tick and retry once
                        for _ in range(40):
                            world.tick()
                        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                    collision_sensor = world.spawn_actor(
                        collision_bp, carla.Transform(), attach_to=vehicle)
                    collision_sensor.listen(on_collision)

                    escorts = spawn_escorts(
                        world, tm, blueprint_library, spawn_points, spawn_point, escort_rng)
                    for _ in range(10):   # let escorts settle before decisions start
                        world.tick()

                    print(f"Episode {ep+1}/{NUM_EPISODES} | "
                          f"Spawn: ({spawn_point.location.x:.1f}, "
                          f"{spawn_point.location.y:.1f}) | "
                          f"escorts: {len(escorts)}/{ESCORT_PER_EPISODE}", flush=True)

                    rows, ep_speed_t = run_episode(
                        ep + 1, vehicle, world, spectator, writer, rng, collision_state)
                    all_rows.extend(rows)
                    all_speed_t.extend(ep_speed_t)
                    crashes = sum(1 for r in rows if r['outcome'] == 'crash')
                    stats_writer.writerow({
                        'episode_id': ep + 1,
                        'spawn_x': round(spawn_point.location.x, 4),
                        'spawn_y': round(spawn_point.location.y, 4),
                        'pairs_logged': len(rows), 'crashes': crashes,
                        'transit_collisions': collision_state['total_hits'],
                        'escorts_spawned': len(escorts),
                        'status': 'ok',
                    })
                    print(f"  Episode {ep+1} complete | pairs: {len(rows)} | "
                          f"crashes: {crashes} | "
                          f"collisions(any): {collision_state['total_hits']}", flush=True)

                except Exception as e:
                    error_msg = f"Episode {ep+1} FAILED: {type(e).__name__}: {e}"
                    print(f"  {error_msg}", flush=True)
                    log_error(error_msg)
                    failed_episodes.append(ep + 1)
                    stats_writer.writerow({
                        'episode_id': ep + 1,
                        'spawn_x': round(spawn_point.location.x, 4),
                        'spawn_y': round(spawn_point.location.y, 4),
                        'pairs_logged': None, 'crashes': None,
                        'transit_collisions': collision_state['total_hits'],
                        'escorts_spawned': len(escorts),
                        'status': 'exception',
                    })

                finally:
                    if collision_sensor is not None:
                        collision_sensor.stop()
                        collision_sensor.destroy()
                    if vehicle is not None:
                        vehicle.set_autopilot(False)
                        vehicle.destroy()
                    destroy_escorts(escorts)
                    for _ in range(10):
                        world.tick()
                    for f in (csvfile, statsfile):
                        f.flush()
                        os.fsync(f.fileno())

        # ── Pilot gates ──────────────────────────────────────────────────────
        n = len(all_rows)
        print(f"\nAll done. Pairs: {n} | data: {OUTPUT_FILE}")
        if all_speed_t:
            std_speed = statistics.pstdev(all_speed_t) if len(all_speed_t) > 1 else 0.0
            frac_stop = sum(1 for s in all_speed_t if s < 0.1) / len(all_speed_t)
            print(f"\n── Gate 1: decision-T saturation (unchanged from banked C) ──")
            print(f"  speed_t std = {std_speed:.3f} m/s (need > ~1.5), "
                  f"frac_near_stop = {frac_stop:.3f}")
            log_error(f"SATURATION: std={std_speed:.4f} frac_stop={frac_stop:.4f}")
        if n:
            for side in ('front', 'rear'):
                pres = sum(1 for r in all_rows if r['rss_t1_%s_present' % side]) / n
                margins = [r['rss_t1_%s_margin' % side] for r in all_rows
                           if r['rss_t1_%s_margin' % side] is not None]
                m_std = statistics.pstdev(margins) if len(margins) > 1 else 0.0
                viol  = (sum(1 for m in margins if m < 0) / len(margins)) if margins else 0.0
                print(f"── Gate 2: RSS {side} coverage at T+1 ──")
                print(f"  present: {100*pres:.0f}% of pairs | margin std {m_std:.2f} m | "
                      f"violation rate {100*viol:.0f}%")
                log_error(f"RSS {side}: present={pres:.3f} std={m_std:.3f} viol={viol:.3f}")
            print("  GATE: need enough coverage to analyze — rear_present at T+1 in")
            print("  >= ~30% of pairs is workable (episode-level aggregation tolerates")
            print("  missing pairs). If coverage is far below that, raise N_NPC before")
            print("  full scale — do NOT make ego behavior depend on traffic instead.")
            if heading_stats['checked']:
                hr = heading_stats['rejected'] / heading_stats['checked']
                print(f"── Diagnostic: heading-mismatch rejections (2026-07-21 fix) ──")
                print(f"  {heading_stats['checked']} geometric candidates checked, "
                      f"{heading_stats['rejected']} rejected as oncoming/cross traffic "
                      f"({100*hr:.1f}%).")
                print("  A high rate here is direct evidence the old code was letting")
                print("  wrong-direction traffic through as a fake front/rear neighbor.")
                log_error(f"HEADING: checked={heading_stats['checked']} "
                          f"rejected={heading_stats['rejected']} rate={hr:.3f}")
            for side in ('front', 'rear'):
                per_ep = {}
                for r in all_rows:
                    ep = r['episode_id']
                    per_ep.setdefault(ep, 0)
                    if r['rss_t1_%s_present' % side]:
                        per_ep[ep] += 1
                n_eps      = len(per_ep)
                zero_eps   = sum(1 for v in per_ep.values() if v == 0)
                total_hits = sum(per_ep.values())
                max_share  = (max(per_ep.values()) / total_hits) if total_hits else 0.0
                print(f"── Gate 4: {side} coverage distribution across episodes ──")
                print(f"  {zero_eps}/{n_eps} episodes have ZERO {side}-covered pairs; "
                      f"busiest single episode holds {100*max_share:.0f}% of all "
                      f"{side} hits.")
                print("  If zero-coverage episodes or max_share are high, coverage is")
                print("  CLUMPED, not just thin — episode-level aggregation does NOT")
                print("  save you here, most episodes contribute nothing to the test.")
                log_error(f"{side.upper()} DISTRIBUTION: zero_eps={zero_eps}/{n_eps} "
                          f"max_share={max_share:.3f}")
            crashes = sum(1 for r in all_rows if r['outcome'] == 'crash')
            print(f"── Gate 3: outcome mix ── crashes {crashes}/{n} "
                  f"({100*crashes/max(n,1):.0f}%)")
            print("  Some crashes are expected and legitimate WITH traffic (a hard")
            print("  scripted brake in front of a follower is the point). If >25% of")
            print("  pairs crash, episodes are dying too early for 8 cycles — consider")
            print("  fewer cycles/episode rather than softening the brake ranges.")
        if failed_episodes:
            summary = f"{len(failed_episodes)}/{NUM_EPISODES} episodes failed: {failed_episodes}"
            print(summary, flush=True)
            log_error(f"RUN SUMMARY: {summary}")
        else:
            log_error(f"RUN SUMMARY: all {NUM_EPISODES} episodes completed cleanly.")

    finally:
        print("Cleaning up NPCs and restoring async mode...")
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
