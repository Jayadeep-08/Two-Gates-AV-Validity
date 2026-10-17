import carla

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    print(f"Connected successfully to CARLA server")
    print(f"Current map: {world.get_map().name}")
    print(f"Number of actors currently in world: {len(world.get_actors())}")

if __name__ == '__main__':
    main()