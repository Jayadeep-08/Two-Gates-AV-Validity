import carla
import time
import math

HOST = 'localhost'
PORT = 2000

client = carla.Client(HOST, PORT)
client.set_timeout(30.0)
world = client.get_world()

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
vehicle_bp.set_attribute('color', '0,0,255')

spawn_points = world.get_map().get_spawn_points()
baseline_spawns = spawn_points[:20]

print(f"Testing {len(baseline_spawns)} spawn points...\n")

good = []
bad  = []

for i, sp in enumerate(baseline_spawns):
    vehicle = None
    try:
        vehicle = world.spawn_actor(vehicle_bp, sp)
        time.sleep(0.5)

        # Check the vehicle is still alive and on a valid road
        transform = vehicle.get_transform()
        waypoint  = world.get_map().get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint is None:
            bad.append((i, sp.location, 'no valid waypoint'))
        else:
            good.append((i, sp.location))
            print(f"  [OK]  #{i} ({sp.location.x:.1f}, {sp.location.y:.1f})")

    except Exception as e:
        bad.append((i, sp.location, str(e)))
        print(f"  [BAD] #{i} ({sp.location.x:.1f}, {sp.location.y:.1f}) → {e}")

    finally:
        if vehicle is not None:
            vehicle.destroy()
        time.sleep(0.3)

print(f"\nResult: {len(good)} good spawn points, {len(bad)} bad")
print(f"Bad ones: {[(b[0], str(b[2])[:60]) for b in bad]}")