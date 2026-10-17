import carla
import time
import csv
import os
import math
import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOST                  = 'localhost'
PORT                  = 2000
TIMEOUT               = 30.0

# ── PILOT MODE ──────────────────────────────────────────────────────────────
# Verification pass, added 2026-07-09: D uses autopilot (Traffic Manager
# handles road-following), so it should NOT have the steering defect that
# broke Scenario C and B — but it never had a collision sensor either, so
# that was an assumption, not a checked fact. Run the 20-episode pilot first
# to confirm 0% crash/off_road before trusting the full 100-episode file.
#
# Pilot passed 2026-07-09 (160/160 safe, clean). Flipped to False 2026-07-11
# — this is the full 100-episode rerun with the collision sensor. The
# existing scenario_d_baseline.csv (800 rows, mean rho 0.0705, dated Jul 1)
# predates this script's collision-sensor rewrite by 8 days and was NOT
# produced by the current script version — that 0.0705 baseline is stale
# and must not be cited anywhere until this run replaces it. The pilot's
# mean (0.0921) already differs by +31%, so expect this run's number to
# differ from 0.0705 too — that's expected, not a bug.
PILOT_MODE            = False
NUM_EPISODES          = 20 if PILOT_MODE else 100
# ──────────────────────────────────────────────────────────────────────────────

DECISIONS_PER_EPISODE = 8
TICKS_PER_DECISION    = 30      # 60 server ticks @ ~20fps = ~3 seconds of sim time
WARMUP_TICKS          = 40      # ticks to let vehicle reach cruising speed
OUTPUT_DIR            = r'C:\Users\jayad\Desktop\RPI_AINN\data'
OUTPUT_FILE           = os.path.join(
    OUTPUT_DIR,
    'scenario_d_baseline_pilot.csv' if PILOT_MODE else 'scenario_d_baseline.csv'
)
SCENARIO_TYPE         = 'D_baseline'
WEATHER               = 'clear'
NPC_DENSITY           = 0
ERROR_LOG_FILE        = OUTPUT_FILE.replace('.csv', '_errors.log')
# ──────────────────────────────────────────────────────────────────────────────


def log_error(msg):
    """Append a timestamped error to a persistent log file — terminal
    scrollback is unreliable for long unattended runs (see scenario_c_braking.py
    for why this was added: it's how 26/400 episodes' failure reasons got lost
    in the 2026-07-09 full run)."""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")


def get_risk_score(vehicle, world):
    """
    Normalized risk score in [0.0, 1.0].
    Combines speed (40%) and lane deviation (60%).
    """
    velocity   = vehicle.get_velocity()
    speed_ms   = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    speed_norm = min(speed_ms / 30.0, 1.0)

    transform  = vehicle.get_transform()
    waypoint   = world.get_map().get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        lane_deviation = 1.0
    else:
        wp_loc    = waypoint.transform.location
        dist      = math.sqrt(
            (transform.location.x - wp_loc.x)**2 +
            (transform.location.y - wp_loc.y)**2
        )
        lane_deviation = min(dist / 3.0, 1.0)

    return round(0.4 * speed_norm + 0.6 * lane_deviation, 6)


def get_state(vehicle, world):
    """Snapshot of vehicle state at this tick."""
    transform = vehicle.get_transform()
    velocity  = vehicle.get_velocity()
    speed_ms  = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    waypoint  = world.get_map().get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    return {
        'x'       : round(transform.location.x, 4),
        'y'       : round(transform.location.y, 4),
        'z'       : round(transform.location.z, 4),
        'yaw'     : round(transform.rotation.yaw, 4),
        'speed_ms': round(speed_ms, 4),
        'lane_id' : waypoint.lane_id if waypoint else None,
        'road_id' : waypoint.road_id if waypoint else None,
    }


def tick_and_follow(world, vehicle, spectator, n_ticks):
    """
    Advance the simulation n_ticks steps in synchronous mode.
    Moves spectator camera to follow vehicle each tick.
    """
    for _ in range(n_ticks):
        world.tick()

        loc = vehicle.get_transform().location
        yaw = vehicle.get_transform().rotation.yaw
        fwd_x = math.cos(math.radians(yaw))
        fwd_y = math.sin(math.radians(yaw))
        cam   = carla.Location(
            x=loc.x - 8 * fwd_x,
            y=loc.y - 8 * fwd_y,
            z=loc.z + 4
        )
        spectator.set_transform(carla.Transform(
            cam, carla.Rotation(pitch=-15, yaw=yaw)
        ))


def run_episode(episode_id, vehicle, world, spectator, writer, collision_state):
    """
    Run one episode in synchronous mode.
    Records DECISIONS_PER_EPISODE consecutive lane_keep decision pairs.
    outcome is now derived from a real CARLA collision sensor plus an
    off-road heuristic instead of being hardcoded to 'safe' — see main() for
    sensor setup. Autopilot (Traffic Manager) drives the whole episode, so we
    do not expect crashes here, but this is now checked, not assumed.
    """
    vehicle.set_autopilot(True)

    # Warmup: let vehicle reach cruising speed
    tick_and_follow(world, vehicle, spectator, WARMUP_TICKS)

    states      = []
    risk_scores = []
    outcomes    = []

    # Sample state at each decision boundary
    for d in range(DECISIONS_PER_EPISODE + 1):
        collision_state['hit'] = False   # reset before this decision window
        states.append(get_state(vehicle, world))
        risk_scores.append(get_risk_score(vehicle, world))

        if d < DECISIONS_PER_EPISODE:
            tick_and_follow(world, vehicle, spectator, TICKS_PER_DECISION)

        # Off-road heuristic: large lane deviation, same pattern as Scenario C
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

        if collision_state['hit']:
            outcomes.append('crash')
        elif off_road:
            outcomes.append('off_road')
        else:
            outcomes.append('safe')

    # Write one row per consecutive pair (t → t+1)
    for t in range(DECISIONS_PER_EPISODE):
        s_t  = states[t]
        s_t1 = states[t + 1]
        writer.writerow({
            'episode_id'  : episode_id,
            't'           : t,
            'x_t'         : s_t['x'],
            'y_t'         : s_t['y'],
            'speed_t'     : s_t['speed_ms'],
            'yaw_t'       : s_t['yaw'],
            'lane_id_t'   : s_t['lane_id'],
            'road_id_t'   : s_t['road_id'],
            'action_t'    : 'lane_keep',
            'x_t1'        : s_t1['x'],
            'y_t1'        : s_t1['y'],
            'speed_t1'    : s_t1['speed_ms'],
            'yaw_t1'      : s_t1['yaw'],
            'lane_id_t1'  : s_t1['lane_id'],
            'road_id_t1'  : s_t1['road_id'],
            'rho_actual'  : risk_scores[t + 1],
            'rho_baseline': None,
            'RPI_pair'    : None,
            'scenario'    : SCENARIO_TYPE,
            'weather'     : WEATHER,
            'npc_density' : NPC_DENSITY,
            'outcome'     : outcomes[t + 1],
        })


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT)
    world  = client.get_world()
    print(f"Connected. Map: {world.get_map().name}")

    # ── Switch to synchronous mode ──────────────────────────────────────────
    settings                     = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05   # 20 ticks per second
    world.apply_settings(settings)
    print("Synchronous mode enabled (20 ticks/sec)")
    # ────────────────────────────────────────────────────────────────────────

    blueprint_library = world.get_blueprint_library()
    vehicle_bp        = blueprint_library.filter('vehicle.tesla.model3')[0]
    vehicle_bp.set_attribute('color', '0,0,255')
    collision_bp      = blueprint_library.find('sensor.other.collision')

    spawn_points    = world.get_map().get_spawn_points()
    baseline_spawns = spawn_points[:20]
    spectator       = world.get_spectator()

    fieldnames = [
        'episode_id', 't',
        'x_t', 'y_t', 'speed_t', 'yaw_t', 'lane_id_t', 'road_id_t',
        'action_t',
        'x_t1', 'y_t1', 'speed_t1', 'yaw_t1', 'lane_id_t1', 'road_id_t1',
        'rho_actual', 'rho_baseline', 'RPI_pair',
        'scenario', 'weather', 'npc_density', 'outcome',
    ]

    failed_episodes = []

    try:
        with open(OUTPUT_FILE, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for ep in range(NUM_EPISODES):
                spawn_point      = baseline_spawns[ep % len(baseline_spawns)]
                vehicle          = None
                collision_sensor = None
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

                    run_episode(ep + 1, vehicle, world, spectator, writer,
                                collision_state)
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
                    # Tick a few times between episodes to let physics settle
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

    finally:
        # ── CRITICAL: always restore async mode before exiting ──────────────
        print("Restoring asynchronous mode...")
        settings                  = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Done.")


if __name__ == '__main__':
    main()