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

# ── PILOT MODE ──────────────────────────────────────────────────────────────
# Flip this to False only after the pilot run's sanity checks pass (see the
# printed distribution summary at the end of the run, and re-check with
# analyse_ablation_diagnostic.py / a fresh describe() on the CSV).
#
# Reset to True on 2026-07-11 — the ranges below were just widened (see the
# BRAKE_INTENSITY_RANGE/BRAKE_DURATION_RANGE comment) and this has NOT been
# piloted yet. Do not flip to False and run 400 episodes on an unpiloted
# range change; that's exactly the mistake that produced the saturated
# original dataset. Gate on the printed sanity summary at the end of the
# pilot run: std(speed_t) should be > ~1.5 m/s (was ~0.53 on the old,
# saturated ranges) and the fraction of pairs with speed_t < 0.1 m/s should
# be well under the old 97.5%.
#
# Pilot passed decisively 2026-07-12: std(speed_t)=3.05 m/s, frac_near_stop=
# 0.181 (both well past gate). More importantly, the actual point of this
# fix — restoring the severity->risk correlation — worked: episode-level
# r(intensity*duration, rho_actual)=0.856, permutation p=0.0 (0/10000),
# vs. the old saturated data's borderline r=0.098/p=0.05. Verified
# independently against the raw pilot CSV, not just the printed summary.
# Flipped to False — this is the full 400-episode rerun.
PILOT_MODE            = False
NUM_EPISODES          = 20 if PILOT_MODE else 400
# ──────────────────────────────────────────────────────────────────────────────

DECISIONS_PER_EPISODE = 8
TICKS_CRUISE          = 40   # ticks at cruise speed before each brake event
# TICKS_RECOVER was 30 (1.5s) — long enough for the vehicle to accelerate back
# to ~cruise speed regardless of how hard/long it braked, which erased any
# correlation between brake severity and the state we sample at T+1 (verified:
# r(brake_intensity_t, rho_actual) = -0.07, r(brake_duration_ticks_t,
# rho_actual) = 0.18 on the 2026-07-09 pilot — both near zero). Shortened to
# 10 ticks (0.5s) so T+1 is sampled while the recovery is still in progress,
# not after it's had time to normalize away the thing we're trying to measure.
TICKS_RECOVER         = 10   # ticks of recovery after brake (decision T+1 window)
WARMUP_TICKS          = 50   # ticks to reach cruising speed before logging starts
CRUISE_THROTTLE       = 0.5  # throttle during cruise phase

# ── Decision T now has real per-episode variance ─────────────────────────────
# Previously BRAKE_INTENSITY=1.0 and TICKS_BRAKE=20 were fixed constants, which
# meant every single brake event was physically identical (speed_t std ≈ 0.0004
# across all 3200 rows in the original run — verified in
# analyse_ablation_diagnostic.py). That made it impossible to test whether the
# SEVERITY of decision T predicts the risk of T+1, because there was no
# variation in severity to correlate against. Sampling intensity and duration
# per brake event fixes this: it gives a continuous "how hard was T" feature
# (logged below as brake_intensity_t / brake_duration_ticks_t) that a future
# permutation/correlation ablation can actually test against rho_actual at T+1.
#
# Widened 2026-07-11 — the first range, (0.5,1.0)/(12,28), still saturated:
# from a ~7 m/s cruise it drove 97.5% of brake events to a complete stop
# (speed_t < 0.1 m/s) regardless of the sampled severity, so there was
# nothing left at decision T to distinguish "hard braking" from "very hard
# braking" — the state was identical either way. Confirmed empirically
# (independently verified against the raw CSV): the 80 pairs that DIDN'T
# fully saturate showed a strong severity->rho_actual correlation
# (pair-level r=0.57, p=2.3e-7 at speed_t>0.5), proof the hypothesis is
# right and the old range was the problem. New range is lower/wider so a
# meaningful fraction of brake events stop short of a full stop.
#
# Severity metric for analysis is decided a priori, before this pilot runs,
# specifically to avoid picking whichever definition looks best after the
# fact: use brake_intensity_t * brake_duration_ticks_t (a braking-impulse
# proxy) as THE severity feature in any correlation/permutation ablation.
# This is the physically motivated choice (impulse, not just peak force) and
# it is NOT being chosen because it scored best on old data — on the old
# saturated full sample it actually did WORSE than intensity alone
# (episode-level p=0.18 vs p=0.05, because duration is pure noise once the
# vehicle is already stopped), which is itself confirmation that saturation,
# not the metric choice, was the problem. Do not swap to a different
# combination of these two columns after seeing the new data.
BRAKE_INTENSITY_RANGE      = (0.15, 0.9)  # uniform range, sampled per brake event
BRAKE_DURATION_RANGE       = (5, 20)      # ticks, sampled per brake event

OUTPUT_DIR            = r'C:\Users\jayad\Desktop\RPI_AINN\data'
OUTPUT_FILE           = os.path.join(
    OUTPUT_DIR,
    'scenario_c_braking_pilot.csv' if PILOT_MODE else 'scenario_c_braking.csv'
)
SCENARIO_TYPE         = 'C_braking'
WEATHER               = 'clear'
NPC_DENSITY           = 0
RNG_SEED              = 42   # reproducible brake-intensity/duration sampling
ERROR_LOG_FILE        = OUTPUT_FILE.replace('.csv', '_errors.log')
# ──────────────────────────────────────────────────────────────────────────────


def log_error(msg):
    """
    Append a timestamped error to a persistent log file instead of relying
    only on the console. Added 2026-07-09 after the 400-episode run silently
    dropped episodes 375-400 (26 episodes, a contiguous block at the tail —
    likely GPU memory pressure over a long unattended session) and the
    specific error text was lost because it only ever printed to a terminal
    that had already scrolled past / closed by the time we went looking.
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")


def soft_cap(x, scale):
    """
    Smooth saturating transform in [0, 1), replacing hard min(x/scale, 1.0)
    clipping. The original formula's lane_deviation and yaw_risk terms used a
    hard clip, and combined with inverse_speed sitting at ~1.0 whenever the
    vehicle is nearly stopped (which is every brake event), ~25% of Scenario C
    rows were pinned at the 0.999+ ceiling (verified against the original
    scenario_c_braking.csv). A hard-clipped score can't tell a moderately bad
    outcome from a catastrophic one once both hit the cap — this compresses
    variance right where the interesting tail behaviour is. 1 - exp(-x/scale)
    approaches 1 asymptotically instead of hitting it, preserving ordering
    among large deviations instead of flattening them all to the same value.
    """
    return 1.0 - math.exp(-x / scale)


def get_risk_score(vehicle, world):
    """
    Risk score for braking scenario.
    Weights speed drop heavily — a near-stopped vehicle on a road is high risk.
    Components:
      - inverse speed (40%): stopped vehicle = high risk of rear-end
      - lane deviation (40%): braking causes lateral drift
      - yaw instability (20%): rapid speed change = lateral instability
    Returns float in [0.0, 1.0).
    """
    velocity = vehicle.get_velocity()
    speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    # Inverse speed: slow vehicle on road = higher risk.
    # This term is intentionally NOT softened — a genuinely stopped vehicle on
    # a live road really is at (near-)maximum rear-end risk, so pinning near
    # 1.0 here reflects reality rather than a formula artifact.
    inverse_speed = max(0.0, 1.0 - speed_ms / 15.0)

    transform = vehicle.get_transform()
    waypoint  = world.get_map().get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        lane_deviation = 1.0
    else:
        wp_loc = waypoint.transform.location
        dist   = math.sqrt(
            (transform.location.x - wp_loc.x)**2 +
            (transform.location.y - wp_loc.y)**2
        )
        lane_deviation = soft_cap(dist, scale=3.0)

    # Yaw deviation from waypoint heading = lateral instability during brake
    if waypoint is not None:
        yaw_diff = abs(transform.rotation.yaw - waypoint.transform.rotation.yaw)
        yaw_diff = min(yaw_diff, 360 - yaw_diff)   # handle wrap-around
        yaw_risk = soft_cap(yaw_diff, scale=30.0)
    else:
        yaw_risk = 1.0

    risk = 0.4 * inverse_speed + 0.4 * lane_deviation + 0.2 * yaw_risk
    return round(min(risk, 0.999999), 6)


def get_state(vehicle, world):
    """Full vehicle state snapshot."""
    transform = vehicle.get_transform()
    velocity  = vehicle.get_velocity()
    speed_ms  = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    waypoint  = world.get_map().get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    return {
        'x'        : round(transform.location.x, 4),
        'y'        : round(transform.location.y, 4),
        'z'        : round(transform.location.z, 4),
        'yaw'      : round(transform.rotation.yaw, 4),
        'speed_ms' : round(speed_ms, 4),
        'lane_id'  : waypoint.lane_id if waypoint else None,
        'road_id'  : waypoint.road_id if waypoint else None,
    }


def apply_control(vehicle, throttle, brake, steer=0.0):
    """Apply direct vehicle control."""
    control          = carla.VehicleControl()
    control.throttle = throttle
    control.brake    = brake
    control.steer    = steer
    vehicle.apply_control(control)


LANE_FOLLOW_LOOKAHEAD = 6.0   # metres ahead to aim steering at, each tick


def lane_follow_steer(vehicle, world, lookahead=LANE_FOLLOW_LOOKAHEAD):
    """
    Compute a steering correction toward the lane centreline ~lookahead metres
    ahead of the vehicle's CURRENT position, recomputed fresh every tick.

    Added 2026-07-09 after the pilot run showed the vehicle driving dead
    straight (steer was hardcoded to 0.0 for the entire cruise/brake/recover
    sequence) and slamming into curbs/planters/trees on any spawn point where
    the road wasn't straight for the ~100m an episode covers — confirmed via
    the collision sensor: 73/160 pilot pairs were 'crash', heavily
    concentrated at specific spawn locations (75-88% crash rate at several,
    0% at others). This mirrors the exact defect that broke Scenario B
    (blind manual control, no real road-following). Fixes it the same way:
    track the lane centreline continuously instead of assuming straight roads.

    Returns steer in [-1, 1], or 0.0 if no waypoint is found (shouldn't happen
    on a real road, but fail safe rather than throw).
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


def tick_and_follow(world, vehicle, spectator, n_ticks,
                    throttle=None, brake=None):
    """
    Advance n_ticks. Optionally hold a fixed throttle/brake each tick, with
    steering computed fresh every tick via lane_follow_steer() so the vehicle
    tracks the road instead of driving straight through curves and off-road
    obstacles. Moves spectator camera to follow vehicle.
    """
    for _ in range(n_ticks):
        if throttle is not None:
            steer = lane_follow_steer(vehicle, world)
            apply_control(vehicle, throttle=throttle,
                          brake=brake if brake else 0.0, steer=steer)
        world.tick()

        loc   = vehicle.get_transform().location
        yaw   = vehicle.get_transform().rotation.yaw
        fwd_x = math.cos(math.radians(yaw))
        fwd_y = math.sin(math.radians(yaw))
        spectator.set_transform(carla.Transform(
            carla.Location(
                x=loc.x - 10 * fwd_x,
                y=loc.y - 10 * fwd_y,
                z=loc.z + 5
            ),
            carla.Rotation(pitch=-20, yaw=yaw)
        ))


def run_episode(episode_id, vehicle, world, spectator, writer, rng,
                collision_state):
    """
    One episode: alternating cruise → brake → recover cycles.
    Each brake event = decision T. Brake intensity/duration are now sampled
    per event (see BRAKE_INTENSITY_RANGE / BRAKE_DURATION_RANGE above).
    State sampled at end of brake = s_T.
    State sampled at end of recover = s_T+1.
    outcome is now derived from a real CARLA collision sensor instead of being
    hardcoded to 'safe' — see main() for sensor setup.
    """
    # Warmup: drive at cruise speed before any logging
    tick_and_follow(world, vehicle, spectator,
                    WARMUP_TICKS, throttle=CRUISE_THROTTLE, brake=0.0)

    states       = []
    risk_scores  = []
    intensities  = []
    durations    = []
    outcomes     = []

    for d in range(DECISIONS_PER_EPISODE):
        # Phase 1: cruise to build speed before brake
        tick_and_follow(world, vehicle, spectator,
                        TICKS_CRUISE, throttle=CRUISE_THROTTLE, brake=0.0)

        # Phase 2: HARD BRAKE — this is decision T. Intensity/duration sampled
        # fresh for this specific brake event, not fixed across the dataset.
        brake_intensity = round(rng.uniform(*BRAKE_INTENSITY_RANGE), 3)
        brake_ticks     = rng.randint(*BRAKE_DURATION_RANGE)

        collision_state['hit'] = False   # reset before the risky phase
        tick_and_follow(world, vehicle, spectator,
                        brake_ticks, throttle=0.0, brake=brake_intensity)

        # Sample state at end of brake (= state s_T, after decision T)
        s_t    = get_state(vehicle, world)
        rho_t  = get_risk_score(vehicle, world)
        states.append(s_t)
        risk_scores.append(rho_t)
        intensities.append(brake_intensity)
        durations.append(brake_ticks)

        # Phase 3: RECOVERY — this is decision T+1
        # Vehicle accelerates back to cruise after the brake
        tick_and_follow(world, vehicle, spectator,
                        TICKS_RECOVER, throttle=CRUISE_THROTTLE, brake=0.0)

        # Sample state at end of recovery (= state s_T+1)
        s_t1   = get_state(vehicle, world)
        rho_t1 = get_risk_score(vehicle, world)
        states.append(s_t1)
        risk_scores.append(rho_t1)

        # Off-road heuristic: get_waypoint with project_to_road=True almost
        # always snaps to a nearby road, so "waypoint is None" rarely fires.
        # A large lane deviation is a better off-road signal in practice.
        transform = vehicle.get_transform()
        waypoint  = world.get_map().get_waypoint(
            transform.location, project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        off_road = False
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
        outcomes.append(outcome)

    # Write decision pairs: each hard_brake (T) → accelerate (T+1) pair
    # DECISIONS_PER_EPISODE=8 brake-recover cycles = 8 T→T+1 pairs per episode
    pair_idx = 0
    for d in range(0, len(states) - 1, 2):
        s_t   = states[d]
        s_t1  = states[d + 1]
        cycle = d // 2
        writer.writerow({
            'episode_id'            : episode_id,
            't'                     : pair_idx,
            'x_t'                   : s_t['x'],
            'y_t'                   : s_t['y'],
            'speed_t'               : s_t['speed_ms'],
            'yaw_t'                 : s_t['yaw'],
            'lane_id_t'             : s_t['lane_id'],
            'road_id_t'             : s_t['road_id'],
            'action_t'              : 'hard_brake',
            'brake_intensity_t'     : intensities[cycle],
            'brake_duration_ticks_t': durations[cycle],
            'x_t1'                  : s_t1['x'],
            'y_t1'                  : s_t1['y'],
            'speed_t1'              : s_t1['speed_ms'],
            'yaw_t1'                : s_t1['yaw'],
            'lane_id_t1'            : s_t1['lane_id'],
            'road_id_t1'            : s_t1['road_id'],
            'rho_actual'            : risk_scores[d + 1],
            'rho_baseline'          : None,
            'RPI_pair'              : None,
            'scenario'              : SCENARIO_TYPE,
            'weather'               : WEATHER,
            'npc_density'           : NPC_DENSITY,
            'outcome'               : outcomes[cycle],
        })
        pair_idx += 1

    # Return decision-T speed_t values so main() can print a sanity summary
    # at the end of the run — the pilot gate (std(speed_t) > ~1.5 m/s, see
    # the PILOT_MODE comment above) needs this without a separate script.
    return [states[d]['speed_ms'] for d in range(0, len(states) - 1, 2)]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = random.Random(RNG_SEED)

    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT)
    world  = client.get_world()
    print(f"Connected. Map: {world.get_map().name}")
    print(f"PILOT_MODE = {PILOT_MODE}  (NUM_EPISODES = {NUM_EPISODES})")
    if PILOT_MODE:
        print("Writing to scenario_c_braking_pilot.csv — this will NOT "
              "overwrite your existing full dataset.")

    # Synchronous mode
    settings                     = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    print("Synchronous mode enabled (20 ticks/sec)")

    blueprint_library = world.get_blueprint_library()
    vehicle_bp        = blueprint_library.filter('vehicle.tesla.model3')[0]
    vehicle_bp.set_attribute('color', '255,50,50')   # red so easy to spot
    collision_bp       = blueprint_library.find('sensor.other.collision')

    spawn_points = world.get_map().get_spawn_points()
    # Use spawn points known-good from diagnosis
    braking_spawns = spawn_points[:20]
    spectator      = world.get_spectator()

    fieldnames = [
        'episode_id', 't',
        'x_t', 'y_t', 'speed_t', 'yaw_t', 'lane_id_t', 'road_id_t',
        'action_t', 'brake_intensity_t', 'brake_duration_ticks_t',
        'x_t1', 'y_t1', 'speed_t1', 'yaw_t1', 'lane_id_t1', 'road_id_t1',
        'rho_actual', 'rho_baseline', 'RPI_pair',
        'scenario', 'weather', 'npc_density', 'outcome',
    ]

    failed_episodes  = []
    all_speed_t      = []   # for the end-of-run saturation sanity check

    try:
        with open(OUTPUT_FILE, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for ep in range(NUM_EPISODES):
                spawn_point = braking_spawns[ep % len(braking_spawns)]
                vehicle          = None
                collision_sensor = None
                # Mutable dict so the collision callback (a closure) can flip
                # a flag the main loop reads — CARLA's sensor callback runs
                # asynchronously and can't return a value directly.
                collision_state  = {'hit': False}

                def on_collision(event):
                    collision_state['hit'] = True

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

                    ep_speed_t = run_episode(ep + 1, vehicle, world, spectator,
                                              writer, rng, collision_state)
                    all_speed_t.extend(ep_speed_t)
                    print(f"  Episode {ep+1} complete", flush=True)

                except Exception as e:
                    error_msg = f"Episode {ep+1} FAILED: {type(e).__name__}: {e}"
                    print(f"  {error_msg}", flush=True)
                    log_error(error_msg)
                    failed_episodes.append(ep + 1)

                finally:
                    if collision_sensor is not None:
                        collision_sensor.stop()
                        collision_sensor.destroy()
                    if vehicle is not None:
                        vehicle.set_autopilot(False)
                        vehicle.destroy()
                    for _ in range(10):
                        world.tick()

        print(f"\nAll done. Data saved to: {OUTPUT_FILE}")
        if failed_episodes:
            summary = (f"{len(failed_episodes)}/{NUM_EPISODES} episodes failed: "
                       f"{failed_episodes}")
            print(summary, flush=True)
            log_error(f"RUN SUMMARY: {summary}")
        else:
            log_error(f"RUN SUMMARY: all {NUM_EPISODES} episodes completed cleanly.")

        # ── Saturation sanity check (2026-07-11) ────────────────────────────
        # Gate before trusting this run / flipping PILOT_MODE to False:
        # std(speed_t) should be well above the old saturated run's ~0.53,
        # target >~1.5 m/s, and frac_near_stop should be well under the old
        # 97.5%. If this still looks saturated, widen the ranges further
        # before running at full scale — don't just flip PILOT_MODE blind.
        if all_speed_t:
            mean_speed = statistics.mean(all_speed_t)
            std_speed  = statistics.pstdev(all_speed_t)
            frac_near_stop = sum(1 for s in all_speed_t if s < 0.1) / len(all_speed_t)
            print(f"\n── Decision-T saturation check ({len(all_speed_t)} pairs) ──")
            print(f"  speed_t mean = {mean_speed:.4f} m/s, std = {std_speed:.4f} m/s")
            print(f"  fraction speed_t < 0.1 m/s (near-full-stop) = {frac_near_stop:.4f}")
            print(f"  GATE: std should be > ~1.5 m/s, frac_near_stop << 0.975 (old value)")
            if std_speed < 1.5 or frac_near_stop > 0.5:
                print("  ⚠ Still looks saturated — widen BRAKE_INTENSITY_RANGE/"
                      "BRAKE_DURATION_RANGE further before running at full scale.")
            else:
                print("  Looks unsaturated — safe to consider flipping PILOT_MODE=False.")
            log_error(f"SATURATION CHECK: mean={mean_speed:.4f} std={std_speed:.4f} "
                      f"frac_near_stop={frac_near_stop:.4f}")

    finally:
        print("Restoring asynchronous mode...")
        settings                  = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Done.")


if __name__ == '__main__':
    main()
