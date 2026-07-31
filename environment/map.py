"""
Road network definition for the Urban Noise Inspection Environment.

This module defines:
- Zone types
- Inspection zones
- Kigali road network
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class ZoneType(Enum):
    """Types of inspection zones."""

    RESIDENTIAL = "Residential"
    COMMERCIAL = "Commercial"
    INDUSTRIAL = "Industrial"
    ENTERTAINMENT = "Entertainment"
    WORSHIP = "Worship"


@dataclass
class InspectionZone:
    """
    Represents a location that may be inspected.
    """

    id: int
    name: str
    zone_type: ZoneType
    risk_probability: float
    x: float
    y: float

    visited: bool = False
    violation_detected: bool = False

    neighbours: List[int] = field(default_factory=list)


class RoadNetwork:
    """
    Graph representation of the inspection network.
    """

    def __init__(self):
        self.zones: Dict[int, InspectionZone] = {}

        self._create_zones()
        self._connect_zones()

    def _create_zones(self):
        """Create Kigali inspection zones."""

        self.zones = {

            0: InspectionZone(
                0,
                "Kacyiru",
                ZoneType.RESIDENTIAL,
                0.20,
                4,
                9,
            ),

            1: InspectionZone(
                1,
                "Kimihurura",
                ZoneType.COMMERCIAL,
                0.45,
                6,
                8,
            ),

            2: InspectionZone(
                2,
                "Remera",
                ZoneType.COMMERCIAL,
                0.50,
                8,
                7,
            ),

            3: InspectionZone(
                3,
                "Amahoro",
                ZoneType.ENTERTAINMENT,
                0.85,
                9,
                5,
            ),

            4: InspectionZone(
                4,
                "Kimironko",
                ZoneType.COMMERCIAL,
                0.60,
                10,
                3,
            ),

            5: InspectionZone(
                5,
                "Kanombe",
                ZoneType.RESIDENTIAL,
                0.25,
                12,
                1,
            ),

            6: InspectionZone(
                6,
                "Gikondo",
                ZoneType.INDUSTRIAL,
                0.75,
                4,
                3,
            ),

            7: InspectionZone(
                7,
                "Nyamirambo",
                ZoneType.ENTERTAINMENT,
                0.80,
                2,
                2,
            ),

            8: InspectionZone(
                8,
                "Nyarugenge",
                ZoneType.COMMERCIAL,
                0.55,
                1,
                5,
            ),

            9: InspectionZone(
                9,
                "Kimisagara",
                ZoneType.WORSHIP,
                0.30,
                0,
                7,
            ),

        }

    def _connect_zones(self):
        """Create road connections."""

        connections = {

            0: [1, 9],

            1: [0, 2, 6],

            2: [1, 3],

            3: [2, 4],

            4: [3, 5],

            5: [4],

            6: [1, 7],

            7: [6, 8],

            8: [7, 9],

            9: [8, 0],

        }

        for zone_id, neighbours in connections.items():
            self.zones[zone_id].neighbours = neighbours

    def get_zone(self, zone_id: int) -> InspectionZone:
        """Return a zone."""

        return self.zones[zone_id]

    def get_neighbours(self, zone_id: int) -> List[int]:
        """Return connected zones."""

        return self.zones[zone_id].neighbours

    def number_of_zones(self) -> int:
        """Return total zones."""

        return len(self.zones)

    def reset(self):
        """Reset environment state."""

        for zone in self.zones.values():

            zone.visited = False
            zone.violation_detected = False

    def __repr__(self):

        return f"RoadNetwork(zones={len(self.zones)})"
    