import carla
import time

client = carla.Client('localhost', 2000)
client.set_timeout(30.0)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

bp = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
spawns = world.get_map().get_spawn_points()

try:
    for i in range(10):
        v = world.spawn_actor(bp, spawns[i % 20])
        v.set_autopilot(True)
        print(f"Episode {i+1}: spawned", flush=True)

        for tick in range(20):
            world.tick()

        v.set_autopilot(False)
        v.destroy()
        print(f"Episode {i+1}: destroyed", flush=True)

        for _ in range(5):
            world.tick()

finally:
    settings.synchronous_mode = False
    world.apply_settings(settings)
    print("Async restored")