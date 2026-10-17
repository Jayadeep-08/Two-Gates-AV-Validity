import carla
import time

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[0]

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Spawned vehicle: {vehicle.type_id} at {spawn_point.location}")

    vehicle.set_autopilot(True)
    print("Autopilot enabled. Vehicle will drive for 15 seconds.")
    print("Watch the CARLA window now.")

    spectator = world.get_spectator()

    for i in range(15):
        transform = vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation

        # Position camera behind and above the vehicle, looking at it
        cam_location = carla.Location(
            x=location.x - 8 * rotation.get_forward_vector().x,
            y=location.y - 8 * rotation.get_forward_vector().y,
            z=location.z + 4
        )
        cam_rotation = carla.Rotation(pitch=-15, yaw=rotation.yaw)
        spectator.set_transform(carla.Transform(cam_location, cam_rotation))

        velocity = vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
        print(f"t={i}s | location=({location.x:.1f}, {location.y:.1f}) | speed={speed:.2f} m/s")
        time.sleep(1)

    vehicle.destroy()
    print("Vehicle destroyed. Test complete.")

if __name__ == '__main__':
    main()