"""
Visualization renderer for Kigali Noise Inspection Environment.

Uses pygame to display:
- Inspection zones
- Roads
- Agent position
- Inspected locations
- Detected violations
"""

import pygame

from environment.map import RoadNetwork



WIDTH = 800
HEIGHT = 600

BACKGROUND = (240, 240, 240)

ROAD_COLOR = (100, 100, 100)

ZONE_COLOR = (50, 150, 255)

AGENT_COLOR = (255, 50, 50)

INSPECTED_COLOR = (50, 200, 50)

VIOLATION_COLOR = (255, 165, 0)



class NoiseRenderer:


    def __init__(self):

        pygame.init()


        self.screen = pygame.display.set_mode(

            (WIDTH, HEIGHT)

        )


        pygame.display.set_caption(

            "Kigali Urban Noise Inspection RL Simulation"

        )


        self.clock = pygame.time.Clock()


        self.network = RoadNetwork()



    def scale_position(
        self,
        x,
        y
    ):

        return (

            int(x * 50 + 100),

            int(HEIGHT - (y * 50 + 100))

        )



    def draw_network(
        self,
        current_zone=None
    ):


        self.screen.fill(

            BACKGROUND

        )


        # Draw roads

        for zone in self.network.zones.values():

            start = self.scale_position(

                zone.x,

                zone.y

            )


            for neighbour in zone.neighbours:


                target_zone = self.network.get_zone(

                    neighbour

                )


                end = self.scale_position(

                    target_zone.x,

                    target_zone.y

                )


                pygame.draw.line(

                    self.screen,

                    ROAD_COLOR,

                    start,

                    end,

                    3

                )



        # Draw zones

        for zone in self.network.zones.values():


            position = self.scale_position(

                zone.x,

                zone.y

            )


            color = ZONE_COLOR



            if zone.visited:

                color = INSPECTED_COLOR



            if zone.violation_detected:

                color = VIOLATION_COLOR



            pygame.draw.circle(

                self.screen,

                color,

                position,

                18

            )



            font = pygame.font.Font(

                None,

                18

            )


            label = font.render(

                zone.name,

                True,

                (0,0,0)

            )


            self.screen.blit(

                label,

                (

                    position[0]-25,

                    position[1]+22

                )

            )



        # Draw agent

        if current_zone is not None:


            zone = self.network.get_zone(

                current_zone

            )


            position = self.scale_position(

                zone.x,

                zone.y

            )


            pygame.draw.circle(

                self.screen,

                AGENT_COLOR,

                position,

                12

            )



        pygame.display.update()



    def close(self):

        pygame.quit()