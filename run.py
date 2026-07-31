from environment.map import RoadNetwork

network = RoadNetwork()

print(network)

for zone in network.zones.values():
    print(
        zone.id,
        zone.name,
        zone.zone_type.value,
        zone.neighbours
    )