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
# Scenario A (highway lane change), written 2026-07-12 AFTER the C/B post-
# mortems, so every design rule from CLAUDE.md's third and fourth audits is
# built in from episode one rather than retrofitted:
#   1. No step functions, booleans, or flat shelves in any risk component —
#      and every capped/normalized component gets its cap CHECKED in the
#      pilot against the range the environment actually produces (B's
#      junction probe capped out for 47% of pairs; that's a step function
#      in disguise).
#   2. Every risk component logged separately at T and T+1, plus rho_t —
#      the per-component variance gate is the pinning detector that would
#      have caught B's plateau in its first pilot.
#   3. No component optimum sits on the controller's own setpoint
#      (OPTIMAL_SPEED != CRUISE_SPEED).
#   4. T+1 is sampled MID-TRANSIENT (TICKS_STABILISE_SAMPLE=15, not after
#      the controller has driven the error to ~0 — C's TICKS_RECOVER=30
#      erased its own signal exactly this way).
#   5. Decision-T severity is sampled per pair with real range, and the
#      severity metric is locked A PRIORI: steer_mag * steer_ticks (the
#      lateral-impulse analog of C's brake_intensity * duration). The
#      RANGES below are pilot-calibratable; the METRIC is not.
#   6. Every discarded attempt is logged with its sampled severity (when
#      sampled) and location — selection-bias audit from episode one.
#   7. Spawn points are pre-filtered for valid lane-change geometry.
#   8. Failed lane changes are NOT discarded (B's selection lesson,
#      inverted): risk is measured relative to the TARGET lane, so a
#      too-weak steer that never leaves the old lane is a legitimate
#      high-risk outcome (large deviation from target), not a
#      severity-correlated hole in the sample. A `lane_change_completed`
#      column makes completion auditable without gating on it.
PILOT_MODE        = False
NUM_EPISODES      = 20 if PILOT_MODE else 400
# ──────────────────────────────────────────────────────────────────────────────

PAIRS_PER_EPISODE = 8
WARMUP_TICKS      = 80    # reach cruise speed before the first pair
TICKS_SETTLE      = 40    # full re-stabilisation between pairs (not measured)

# ── Decision T: initiate lane change (sampled severity) ──────────────────────
# Severity metric locked a priori: steer_mag * steer_ticks. Physics sanity
# for the starting ranges: at ~8 m/s a CARLA steer of 0.10 gives a yaw rate
# around 15-20 deg/s; one lane (~3.5 m lateral) over ~1 s of steering needs
# roughly that. The readme's original suggestion (0.25-0.6 x 15-35) would be
# a violent swerve at speed — start gentler and let the PILOT calibrate the
# bounds (calibrating ranges is allowed; changing the metric after seeing
# full-scale results is not).
STEER_MAG_RANGE   = (0.05, 0.20)   # uniform, sampled per pair
STEER_TICKS_RANGE = (10, 25)       # uniform integer, sampled per pair

# ── Decision T+1: stabilise in the new lane ──────────────────────────────────
# T+1 is sampled after TICKS_STABILISE_SAMPLE ticks of stabilisation — while
# the lateral/heading transient is still alive (rule 4). Pilot gate: severity
# must correlate with the realized T state; if the transient is already dead
# at 15 ticks, lower this before touching anything else.
TICKS_STABILISE_SAMPLE = 15

CRUISE_SPEED      = 8.0    # m/s controller target during cruise + maneuver
CRUISE_LOOKAHEAD  = 6.0    # metres ahead for lane-follow steering
OPTIMAL_SPEED     = 7.0    # m/s risk-formula optimum — deliberately != CRUISE_SPEED (rule 3)
SPEED_RISK_SCALE  = 8.0    # m/s scale for the speed U-curve
LAT_DEV_SCALE     = 1.5    # metres — soft_cap scale for lateral deviation from TARGET lane centre
HEADING_ERR_SCALE = 20.0   # degrees — soft_cap scale for heading error vs target lane direction

MAX_RELOCATE_TRIES = 3     # cruise-and-retry attempts when no valid adjacent lane exists here
RELOCATE_TICKS     = 50    # cruise ticks between relocation attempts

RNG_SEED          = 42     # reproducible severity sampling

OUTPUT_DIR        = r'C:\Users\jayad\Desktop\RPI_AINN\data'
OUTPUT_FILE       = os.path.join(
    OUTPUT_DIR,
    'scenario_a_lanechange_pilot.csv' if PILOT_MODE else 'scenario_a_lanechange.csv'
)
SCENARIO_TYPE     = 'A_lanechange'
WEATHER           = 'clear'
NPC_DENSITY       = 0
ERROR_LOG_FILE    = OUTPUT_FILE.replace('.csv', '_errors.log')
DISCARD_LOG_FILE   = OUTPUT_FILE.replace('.csv', '_discards.csv')
EPISODE_STATS_FILE = OUTPUT_FILE.replace('.csv', '_episode_stats.csv')
# ──────────────────────────────────────────────────────────────────────────────


def log_error(msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")


def pearson_r(xs, ys):
    """Tiny dependency-free Pearson r for the end-of-run gates."""
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
    """Smooth saturating transform in [0,1) — same as scenario_c_braking.py.
    No hard clip anywhere, per design rule 1."""
    return 1.0 - math.exp(-x / scale)


# ── Target-lane machinery ────────────────────────────────────────────────────
# Risk is measured relative to the lane the maneuver INTENDED to reach
# (rule 8). "Target lane" is resolved dynamically: project the vehicle to
# the road; if the projected lane is still the original lane, the target is
# its adjacent lane in the maneuver direction; otherwise the vehicle has
# crossed over and the projected lane IS the target. This keeps the target
# well-defined even though lane_id/road_id values change along the road.

def adjacent_lane(waypoint, direction):
    """Same-direction Driving neighbour of `waypoint` on `direction` side,
    or None. Same-direction check: OpenDRIVE lane_ids on the same side of
    the road share sign — an opposite-sign neighbour is oncoming traffic
    (design rule 7's failure mode)."""
    nb = waypoint.get_left_lane() if direction == 'left' else waypoint.get_right_lane()
    if nb is None:
        return None
    if nb.lane_type != carla.LaneType.Driving:
        return None
    if nb.lane_id * waypoint.lane_id < 0:
        return None
    return nb


def resolve_target_waypoint(world, vehicle, original_lane_id, direction):
    """Waypoint on the TARGET lane's centreline nearest the vehicle, plus a
    flag for whether the vehicle's own projection already IS the target."""
    wp = world.get_map().get_waypoint(
        vehicle.get_transform().location,
        project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if wp is None:
        return None, False
    if wp.lane_id == original_lane_id:
        return adjacent_lane(wp, direction), False
    return wp, True


def compute_risk_components(vehicle, world, original_lane_id, direction):
    """
    Three continuous components + combined score, all relative to the
    TARGET lane (rule 8). Logged separately at T and T+1 (rule 2).

      - lat_dev  (0.5): soft_cap(distance to target-lane centreline, 1.5m).
        Primary danger axis for a lane change. A failed change (still in
        the old lane) reads ~3.5m -> ~0.9 — high risk, as it should.
      - heading_err (0.3): soft_cap(|yaw - target lane yaw| deg, 20deg).
        Lateral instability / not yet aligned with the new lane.
      - speed_risk (0.2): smooth U-curve around OPTIMAL_SPEED (7.0), which
        is deliberately not the controller target (8.0) — rule 3.
    """
    transform = vehicle.get_transform()
    velocity  = vehicle.get_velocity()
    speed_ms  = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    target_wp, _ = resolve_target_waypoint(world, vehicle, original_lane_id, direction)

    if target_wp is None:
        lat_dev_m   = LAT_DEV_SCALE * 3.0   # no resolvable target — treat as far off
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
        'lat_dev_m'  : round(lat_dev_m, 4),      # raw metres, for the pilot gate
        'heading_deg': round(heading_deg, 4),    # raw degrees, for the pilot gate
    }


def get_risk_score(vehicle, world, original_lane_id, direction):
    return compute_risk_components(vehicle, world, original_lane_id, direction)['rho']


def get_state(vehicle, world):
    transform = vehicle.get_transform()
    velocity  = vehicle.get_velocity()
    speed_ms  = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    waypoint  = world.get_map().get_waypoint(
        transform.location, project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    return {
        'x'        : round(transform.location.x, 4),
        'y'        : round(transform.location.y, 4),
        'yaw'      : round(transform.rotation.yaw, 4),
        'speed_ms' : round(speed_ms, 4),
        'lane_id'  : waypoint.lane_id if waypoint else None,
        'road_id'  : waypoint.road_id if waypoint else None,
    }


def lane_follow_steer(vehicle, world, lookahead):
    """Per-tick steering toward the NEAREST lane centreline (cruise phases)."""
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
    return _steer_toward(vehicle, ahead[0].transform.location)


def target_lane_steer(vehicle, world, original_lane_id, direction, lookahead):
    """Per-tick steering toward the TARGET lane's centreline `lookahead`
    metres ahead — the stabilisation controller. Unlike lane_follow_steer,
    this will not 'stabilise' a half-finished change back into the old lane
    (nearest-lane projection would do exactly that while the vehicle is
    still on the old side of the boundary)."""
    target_wp, _ = resolve_target_waypoint(world, vehicle, original_lane_id, direction)
    if target_wp is None:
        return lane_follow_steer(vehicle, world, lookahead)
    ahead = target_wp.next(lookahead)
    if not ahead:
        return 0.0
    return _steer_toward(vehicle, ahead[0].transform.location)


def _steer_toward(vehicle, target_loc):
    transform   = vehicle.get_transform()
    vehicle_yaw = math.radians(transform.rotation.yaw)
    dx = target_loc.x - transform.location.x
    dy = target_loc.y - transform.location.y
    yaw_diff = math.atan2(dy, dx) - vehicle_yaw
    while yaw_diff >  math.pi: yaw_diff -= 2 * math.pi
    while yaw_diff < -math.pi: yaw_diff += 2 * math.pi
    return max(-1.0, min(1.0, yaw_diff / math.pi * 2.0))


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
    """Steady-state lane keeping at CRUISE_SPEED (warmup / settle / relocate)."""
    for _ in range(n_ticks):
        ctrl          = carla.VehicleControl()
        ctrl.throttle = speed_throttle(vehicle, CRUISE_SPEED)
        ctrl.brake    = 0.0
        ctrl.steer    = lane_follow_steer(vehicle, world, CRUISE_LOOKAHEAD)
        vehicle.apply_control(ctrl)
        tick_world(world, vehicle, spectator)


def execute_steer_phase(world, vehicle, spectator, steer_mag, steer_ticks, direction):
    """Decision T: open-loop steer input toward `direction` for steer_ticks.
    Throttle keeps regulating toward CRUISE_SPEED so severity is purely the
    lateral impulse, not a confounded speed change."""
    steer_sign = -1.0 if direction == 'left' else 1.0   # CARLA: negative steer = left
    for _ in range(steer_ticks):
        ctrl          = carla.VehicleControl()
        ctrl.throttle = speed_throttle(vehicle, CRUISE_SPEED)
        ctrl.brake    = 0.0
        ctrl.steer    = steer_sign * steer_mag
        vehicle.apply_control(ctrl)
        tick_world(world, vehicle, spectator)


def stabilise_phase(world, vehicle, spectator, original_lane_id, direction, n_ticks):
    """Decision T+1: closed-loop stabilisation toward the TARGET lane."""
    for _ in range(n_ticks):
        ctrl          = carla.VehicleControl()
        ctrl.throttle = speed_throttle(vehicle, CRUISE_SPEED)
        ctrl.brake    = 0.0
        ctrl.steer    = target_lane_steer(vehicle, world, original_lane_id,
                                          direction, CRUISE_LOOKAHEAD)
        vehicle.apply_control(ctrl)
        tick_world(world, vehicle, spectator)


def run_episode(episode_id, vehicle, world, spectator, writer, discard_writer,
                collision_state, rng, spawn_point):
    """
    One episode: up to PAIRS_PER_EPISODE lane-change (T) -> stabilise (T+1)
    pairs, alternating left/right so the vehicle walks across and back
    within the road. Failed changes are logged, not discarded (rule 8);
    the only discards are structural (no valid adjacent lane after
    relocation tries, crash, off-road) and every one gets a sidecar row.
    """
    pairs_logged      = 0
    no_lane_fails     = 0
    crashes           = 0
    off_roads         = 0
    severity_values   = []   # steer_mag * steer_ticks, for the gate
    lat_dev_t_values  = []   # raw metres at T, for the severity->state gate
    component_values  = []   # comp_t1 dicts, per-component variance gate
    completed_flags   = []
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
        })

    for pair in range(PAIRS_PER_EPISODE):
        # ── Find a spot with a valid adjacent lane (rule 7, per-pair) ────────
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

        # ── Decision T: initiate lane change (sampled severity, rule 5) ─────
        steer_mag   = round(rng.uniform(*STEER_MAG_RANGE), 3)
        steer_ticks = rng.randint(*STEER_TICKS_RANGE)
        severity    = round(steer_mag * steer_ticks, 4)
        collision_state['hit'] = False

        execute_steer_phase(world, vehicle, spectator, steer_mag, steer_ticks, direction)
        s_t    = get_state(vehicle, world)
        comp_t = compute_risk_components(vehicle, world, original_lane_id, direction)

        # ── Decision T+1: stabilise toward the target lane, sample
        # MID-TRANSIENT (rule 4) ─────────────────────────────────────────────
        stabilise_phase(world, vehicle, spectator, original_lane_id, direction,
                        TICKS_STABILISE_SAMPLE)
        s_t1    = get_state(vehicle, world)
        comp_t1 = compute_risk_components(vehicle, world, original_lane_id, direction)

        completed = (s_t1['lane_id'] == target_lane_id)

        # Outcome from the collision sensor + off-road heuristic
        off_road = comp_t1['lat_dev_m'] > 5.0
        if collision_state['hit']:
            outcome = 'crash'
        elif off_road:
            outcome = 'off_road'
        else:
            outcome = 'safe'

        writer.writerow({
            'episode_id'    : episode_id,
            't'             : pairs_logged,
            'x_t'           : s_t['x'],
            'y_t'           : s_t['y'],
            'speed_t'       : s_t['speed_ms'],
            'yaw_t'         : s_t['yaw'],
            'lane_id_t'     : s_t['lane_id'],
            'road_id_t'     : s_t['road_id'],
            'action_t'      : 'lane_change_' + direction,
            'steer_mag_t'   : steer_mag,
            'steer_ticks_t' : steer_ticks,
            'severity_t'    : severity,
            'lat_dev_t'     : comp_t['lat_dev'],
            'heading_err_t' : comp_t['heading_err'],
            'speed_risk_t'  : comp_t['speed_risk'],
            'lat_dev_m_t'   : comp_t['lat_dev_m'],
            'heading_deg_t' : comp_t['heading_deg'],
            'rho_t'         : comp_t['rho'],
            'x_t1'          : s_t1['x'],
            'y_t1'          : s_t1['y'],
            'speed_t1'      : s_t1['speed_ms'],
            'yaw_t1'        : s_t1['yaw'],
            'lane_id_t1'    : s_t1['lane_id'],
            'road_id_t1'    : s_t1['road_id'],
            'lat_dev_t1'    : comp_t1['lat_dev'],
            'heading_err_t1': comp_t1['heading_err'],
            'speed_risk_t1' : comp_t1['speed_risk'],
            'lat_dev_m_t1'  : comp_t1['lat_dev_m'],
            'heading_deg_t1': comp_t1['heading_deg'],
            'lane_change_completed': completed,
            'rho_actual'    : comp_t1['rho'],
            'rho_baseline'  : None,
            'RPI_pair'      : None,
            'scenario'      : SCENARIO_TYPE,
            'weather'       : WEATHER,
            'npc_density'   : NPC_DENSITY,
            'outcome'       : outcome,
        })
        pairs_logged += 1
        severity_values.append(severity)
        lat_dev_t_values.append(comp_t['lat_dev_m'])
        component_values.append(comp_t1)
        completed_flags.append(completed)

        if outcome == 'crash':
            crashes += 1
            log_discard(pair, 'crash_ends_episode', severity)
            break   # a crashed vehicle's later pairs would be garbage
        if outcome == 'off_road':
            off_roads += 1
            log_discard(pair, 'off_road_ends_episode', severity)
            break

        # ── Settle fully in the (new) lane before the next pair ─────────────
        cruise(world, vehicle, spectator, TICKS_SETTLE)
        direction = 'right' if direction == 'left' else 'left'

    return (pairs_logged, no_lane_fails, crashes, off_roads,
            severity_values, lat_dev_t_values, component_values, completed_flags)


def filter_spawn_points(world, all_spawns):
    """Rule 7: keep only spawn points whose lane has a same-direction
    Driving neighbour on at least one side."""
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
    vehicle_bp.set_attribute('color', '200,120,0')
    collision_bp      = blueprint_library.find('sensor.other.collision')

    all_spawns = world.get_map().get_spawn_points()
    spawns     = filter_spawn_points(world, all_spawns)
    print(f"Spawn filter: {len(spawns)}/{len(all_spawns)} points have a valid "
          f"same-direction adjacent lane")
    if not spawns:
        raise RuntimeError("No lane-change-capable spawn points on this map.")
    spectator = world.get_spectator()

    fieldnames = [
        'episode_id', 't',
        'x_t', 'y_t', 'speed_t', 'yaw_t', 'lane_id_t', 'road_id_t',
        'action_t', 'steer_mag_t', 'steer_ticks_t', 'severity_t',
        'lat_dev_t', 'heading_err_t', 'speed_risk_t',
        'lat_dev_m_t', 'heading_deg_t', 'rho_t',
        'x_t1', 'y_t1', 'speed_t1', 'yaw_t1', 'lane_id_t1', 'road_id_t1',
        'lat_dev_t1', 'heading_err_t1', 'speed_risk_t1',
        'lat_dev_m_t1', 'heading_deg_t1',
        'lane_change_completed',
        'rho_actual', 'rho_baseline', 'RPI_pair',
        'scenario', 'weather', 'npc_density', 'outcome',
    ]
    discard_fieldnames = [
        'episode_id', 'pair_attempt', 'failure_type', 'severity_t',
        'spawn_x', 'spawn_y', 'fail_x', 'fail_y', 'fail_speed',
    ]
    episode_stats_fieldnames = [
        'episode_id', 'spawn_x', 'spawn_y', 'pairs_logged',
        'no_lane_fails', 'crashes', 'off_roads', 'completed_changes',
        'status',
    ]

    rng = random.Random(RNG_SEED)

    total_pairs        = 0
    total_no_lane      = 0
    total_crashes      = 0
    total_off_roads    = 0
    failed_episodes    = []
    all_severity       = []
    all_lat_dev_t      = []
    all_components     = []
    all_completed      = []

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
                spawn_point      = spawns[ep % len(spawns)]
                vehicle          = None
                collision_sensor = None
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
                          f"{spawn_point.location.y:.1f})", flush=True)

                    cruise(world, vehicle, spectator, WARMUP_TICKS)

                    (pairs, no_lane, crashes, off_roads, sev, latm, comps,
                     completed) = run_episode(
                        ep + 1, vehicle, world, spectator,
                        writer, discard_writer, collision_state, rng, spawn_point
                    )
                    total_pairs     += pairs
                    total_no_lane   += no_lane
                    total_crashes   += crashes
                    total_off_roads += off_roads
                    all_severity.extend(sev)
                    all_lat_dev_t.extend(latm)
                    all_components.extend(comps)
                    all_completed.extend(completed)

                    stats_writer.writerow({
                        'episode_id'       : ep + 1,
                        'spawn_x'          : round(spawn_point.location.x, 4),
                        'spawn_y'          : round(spawn_point.location.y, 4),
                        'pairs_logged'     : pairs,
                        'no_lane_fails'    : no_lane,
                        'crashes'          : crashes,
                        'off_roads'        : off_roads,
                        'completed_changes': sum(completed),
                        'status'           : 'ok',
                    })
                    print(f"  Episode {ep+1} complete | pairs: {pairs} | "
                          f"completed changes: {sum(completed)}/{pairs} | "
                          f"no_lane_fails: {no_lane} | crashes: {crashes} | "
                          f"off_roads: {off_roads} | total so far: {total_pairs}",
                          flush=True)

                except Exception as e:
                    error_msg = f"Episode {ep+1} FAILED: {type(e).__name__}: {e}"
                    print(f"  {error_msg}", flush=True)
                    log_error(error_msg)
                    failed_episodes.append(ep + 1)
                    stats_writer.writerow({
                        'episode_id'       : ep + 1,
                        'spawn_x'          : round(spawn_point.location.x, 4),
                        'spawn_y'          : round(spawn_point.location.y, 4),
                        'pairs_logged'     : None,
                        'no_lane_fails'    : None,
                        'crashes'          : None,
                        'off_roads'        : None,
                        'completed_changes': None,
                        'status'           : 'exception',
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
              f"off_roads: {total_off_roads} | "
              f"completed changes: {sum(all_completed)}/{total_pairs}")
        print(f"Data saved to: {OUTPUT_FILE}")
        print(f"Discarded-attempt log: {DISCARD_LOG_FILE}")
        print(f"Per-episode stats: {EPISODE_STATS_FILE}")

        # ── Gate 1: decision-T severity spread + severity -> T-state link ───
        if all_severity:
            sev_std = statistics.pstdev(all_severity) if len(all_severity) > 1 else 0.0
            lat_std = statistics.pstdev(all_lat_dev_t) if len(all_lat_dev_t) > 1 else 0.0
            r_sev_lat = pearson_r(all_severity, all_lat_dev_t)
            print(f"\n── Decision-T severity gate ({len(all_severity)} pairs) ──")
            print(f"  severity (mag*ticks): mean {statistics.mean(all_severity):.3f}, std {sev_std:.3f}")
            print(f"  realized lat_dev at T (m): mean {statistics.mean(all_lat_dev_t):.3f}, std {lat_std:.3f}")
            print(f"  corr(severity, lat_dev_m at T) = {r_sev_lat:+.3f}")
            print(f"  GATE: severity must produce real spread in the T state AND")
            print(f"  correlate with it (rule 5). If corr is ~0, the transient is")
            print(f"  dead before T is sampled — shorten steer ticks range or check")
            print(f"  the controller, don't proceed to full scale.")
            log_error(f"SEVERITY GATE: sev_std={sev_std:.4f} lat_std={lat_std:.4f} r={r_sev_lat:+.4f}")

        # ── Gate 2: per-component variance at T+1 (pinning detector) ────────
        if all_components:
            print(f"\n── Per-component variance check ({len(all_components)} pairs, T+1) ──")
            for key in ('lat_dev', 'heading_err', 'speed_risk'):
                vals = [c[key] for c in all_components]
                mean_v = statistics.mean(vals)
                std_v  = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                near_cap = sum(1 for v in vals if v > 0.95) / len(vals)
                print(f"  {key:12s} mean = {mean_v:.4f}, std = {std_v:.4f}, "
                      f"frac>0.95 = {near_cap:.2f}")
                log_error(f"COMPONENT CHECK {key}: mean={mean_v:.4f} std={std_v:.4f} frac_cap={near_cap:.3f}")
            print(f"  GATE: every component std well above 0 AND frac>0.95 low.")
            print(f"  (B's probe cap silently flattened 47% of its sample — check")
            print(f"  the cap/scale of ANY component that clusters near 1.0.)")

        # ── Gate 3: completion mix ───────────────────────────────────────────
        if all_completed:
            frac = sum(all_completed) / len(all_completed)
            print(f"\n── Completion mix ── {sum(all_completed)}/{len(all_completed)} "
                  f"changes completed ({100*frac:.0f}%)")
            print(f"  Healthy range is roughly 40-90%: near 100% means the severity")
            print(f"  range is so strong every change succeeds (no contrast); near 0%")
            print(f"  means it's too weak to ever change lanes. Calibrate ranges, not")
            print(f"  the metric.")
            log_error(f"COMPLETION MIX: {sum(all_completed)}/{len(all_completed)}")

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
