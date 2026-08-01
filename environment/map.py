"""
Road network definition for the Urban Noise Inspection Environment.

Defines:
- Kigali inspection zones
- Zone categories
- Road connections with distances
- Hidden noise violation states
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List
import random



class ZoneType(Enum):
    """
    Types of urban inspection zones.
    """

    RESIDENTIAL = "Residential"
    COMMERCIAL = "Commercial"
    INDUSTRIAL = "Industrial"
    ENTERTAINMENT = "Entertainment"
    WORSHIP = "Worship"




@dataclass
class InspectionZone:
    """
    Represents a Kigali inspection location.
    """

    id: int
    name: str
    zone_type: ZoneType

    risk_probability: float

    x: float
    y: float

    # Inspection tracking
    visited: bool = False

    inspected: bool = False

    # Noise violation tracking
    violation_detected: bool = False

    has_violation: bool = False

    # Connected roads
    neighbours: List[int] = field(
        default_factory=list
    )




class RoadNetwork:
    """
    Graph representation of Kigali inspection zones.

    Roads store:
        destination_zone: distance

    Distance represents travel cost.
    """

    def __init__(self):

        self.zones: Dict[int, InspectionZone] = {}

        self.roads: Dict[int, Dict[int, float]] = {}

        self._create_zones()

        self._create_roads()

        self._update_neighbours()



    # --------------------------------------------------
    # Create Kigali inspection zones
    # --------------------------------------------------

    def _create_zones(self):

        self.zones = {

            0: InspectionZone(
                0,
                "Kacyiru",
                ZoneType.RESIDENTIAL,
                0.20,
                4,
                9
            ),

            1: InspectionZone(
                1,
                "Kimihurura",
                ZoneType.COMMERCIAL,
                0.45,
                6,
                8
            ),

            2: InspectionZone(
                2,
                "Remera",
                ZoneType.COMMERCIAL,
                0.50,
                8,
                7
            ),

            3: InspectionZone(
                3,
                "Amahoro",
                ZoneType.ENTERTAINMENT,
                0.85,
                9,
                5
            ),

            4: InspectionZone(
                4,
                "Kimironko",
                ZoneType.COMMERCIAL,
                0.60,
                10,
                3
            ),

            5: InspectionZone(
                5,
                "Kanombe",
                ZoneType.RESIDENTIAL,
                0.25,
                12,
                1
            ),

            6: InspectionZone(
                6,
                "Gikondo",
                ZoneType.INDUSTRIAL,
                0.75,
                4,
                3
            ),

            7: InspectionZone(
                7,
                "Nyamirambo",
                ZoneType.ENTERTAINMENT,
                0.80,
                2,
                2
            ),

            8: InspectionZone(
                8,
                "Nyarugenge",
                ZoneType.COMMERCIAL,
                0.55,
                1,
                5
            ),

            9: InspectionZone(
                9,
                "Kimisagara",
                ZoneType.WORSHIP,
                0.30,
                0,
                7
            )

        }



    # --------------------------------------------------
    # Create road network
    # --------------------------------------------------

    def _create_roads(self):

        self.roads = {

            0: {
                1: 3.5,
                9: 4.0
            },

            1: {
                0: 3.5,
                2: 2.8,
                6: 5.2
            },

            2: {
                1: 2.8,
                3: 3.1
            },

            3: {
                2: 3.1,
                4: 2.5
            },

            4: {
                3: 2.5,
                5: 3.8
            },

            5: {
                4: 3.8
            },

            6: {
                1: 5.2,
                7: 2.9
            },

            7: {
                6: 2.9,
                8: 3.0
            },

            8: {
                7: 3.0,
                9: 2.6
            },

            9: {
                8: 2.6,
                0: 4.0
            }

        }



    def _update_neighbours(self):

        for zone_id, connections in self.roads.items():

            self.zones[zone_id].neighbours = list(
                connections.keys()
            )



    # --------------------------------------------------
    # Environment helper functions
    # --------------------------------------------------

    def generate_noise_events(self):

        """
        Generate hidden noise violations.

        Agent discovers them only after inspection.
        """

        for zone in self.zones.values():

            zone.has_violation = (

                random.random()

                < zone.risk_probability

            )



    def inspect_zone(
        self,
        zone_id: int
    ):

        """
        Reveal whether a zone violates noise limits.
        """

        zone = self.zones[zone_id]

        zone.inspected = True

        zone.visited = True

        if zone.has_violation:

            zone.violation_detected = True

        return zone.has_violation




    def get_zone(
        self,
        zone_id: int
    ):

        return self.zones[zone_id]




    def get_connections(
        self,
        zone_id: int
    ):

        return self.roads[zone_id]




    def get_neighbours(
        self,
        zone_id: int
    ):

        return self.zones[zone_id].neighbours




    def get_distance(
        self,
        start_zone: int,
        destination_zone: int
    ):

        return self.roads[start_zone][destination_zone]




    def number_of_zones(self):

        return len(self.zones)




    def reset(self):

        for zone in self.zones.values():

            zone.visited = False

            zone.inspected = False

            zone.violation_detected = False

            zone.has_violation = False




    def __repr__(self):

        return (

            f"RoadNetwork("
            f"zones={len(self.zones)}, "
            f"roads={sum(len(x) for x in self.roads.values())}"
            f")"

        )