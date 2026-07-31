"""
Road network definition for the Urban Noise Inspection Environment.

This module defines:
- Kigali inspection zones
- Zone categories
- Road connections with distances
- Hidden noise violation states
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict
import random


class ZoneType(Enum):
    """Types of urban inspection zones."""

    RESIDENTIAL = "Residential"
    COMMERCIAL = "Commercial"
    INDUSTRIAL = "Industrial"
    ENTERTAINMENT = "Entertainment"
    WORSHIP = "Worship"


@dataclass
class InspectionZone:
    """
    Represents an inspection location.
    """

    id: int
    name: str
    zone_type: ZoneType

    # Probability that this location violates noise regulations
    risk_probability: float

    # Coordinates for visualization
    x: float
    y: float

    # Runtime states
    inspected: bool = False
    has_violation: bool = False


class RoadNetwork:
    """
    Graph representation of Kigali inspection zones.

    Connections are stored as:

    {
        source_zone:
            {
                destination_zone: distance
            }
    }

    Distance represents travel cost.
    """

    def __init__(self):

        self.zones: Dict[int, InspectionZone] = {}

        self.roads: Dict[int, Dict[int, float]] = {}

        self._create_zones()

        self._create_roads()


    # --------------------------------------------------
    # Create Kigali zones
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
            ),

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


    # --------------------------------------------------
    # Environment helpers
    # --------------------------------------------------

    def generate_noise_events(self):
        """
        Generate hidden noise violations.

        The agent does not know these values.
        It must inspect locations to discover them.
        """

        for zone in self.zones.values():

            zone.has_violation = (
                random.random()
                < zone.risk_probability
            )


    def get_zone(self, zone_id: int):

        return self.zones[zone_id]


    def get_connections(self, zone_id: int):

        return self.roads[zone_id]


    def get_distance(
        self,
        start_zone: int,
        destination_zone: int
    ):

        return self.roads[start_zone][destination_zone]


    def number_of_zones(self):

        return len(self.zones)


    def reset(self):
        """
        Reset episode state.
        """

        for zone in self.zones.values():

            zone.inspected = False
            zone.has_violation = False


    def __repr__(self):

        return (
            f"RoadNetwork("
            f"zones={len(self.zones)}, "
            f"roads={sum(len(x) for x in self.roads.values())}"
            f")"
        )