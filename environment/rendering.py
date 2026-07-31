"""
Pygame visualization for Kigali Urban Noise Inspection Environment.
"""

import pygame


class NoiseRenderer:

    def __init__(self):

        pygame.init()

        self.width = 900
        self.height = 700

        self.screen = pygame.display.set_mode(
            (self.width, self.height)
        )

        pygame.display.set_caption(
            "Kigali Urban Noise Inspection Simulation"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "Arial",
            16
        )


        self.scale = 45
        self.offset_x = 100
        self.offset_y = 80



    def _convert_position(self, x, y):

        return (

            int(self.offset_x + x * self.scale),

            int(self.offset_y + y * self.scale)

        )



    def draw_roads(self, env):

        for start, connections in env.network.roads.items():

            start_zone = env.network.get_zone(start)

            start_position = self._convert_position(
                start_zone.x,
                start_zone.y
            )


            for destination in connections:

                end_zone = env.network.get_zone(
                    destination
                )


                end_position = self._convert_position(
                    end_zone.x,
                    end_zone.y
                )


                pygame.draw.line(

                    self.screen,

                    (150,150,150),

                    start_position,

                    end_position,

                    3

                )



    def draw_zones(self, env):

        for zone in env.network.zones.values():

            position = self._convert_position(
                zone.x,
                zone.y
            )


            radius = 20


            if zone.id == env.current_zone:

                color = (0,255,0)


            elif zone.inspected:

                color = (0,150,255)


            else:

                color = (200,200,200)



            pygame.draw.circle(

                self.screen,

                color,

                position,

                radius

            )


            label = self.font.render(

                zone.name,

                True,

                (0,0,0)

            )


            self.screen.blit(

                label,

                (
                    position[0]-30,
                    position[1]+25
                )

            )



            if zone.inspected and zone.has_violation:

                violation = self.font.render(

                    "Noise Violation",

                    True,

                    (255,0,0)

                )

                self.screen.blit(

                    violation,

                    (
                        position[0]-45,
                        position[1]-35
                    )

                )



    def draw_information(self, env):

        info = [

            f"Current Zone: {env.network.get_zone(env.current_zone).name}",

            f"Battery: {env.battery}",

            f"Time: {env.remaining_time}",

            f"Violations Found: {env.violations_found}"

        ]


        y = 20


        for text in info:

            surface = self.font.render(

                text,

                True,

                (0,0,0)

            )

            self.screen.blit(

                surface,

                (20,y)

            )

            y += 25



    def render(self, env):

        running = True


        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False



        self.screen.fill(
            (240,240,240)
        )


        self.draw_roads(env)

        self.draw_zones(env)

        self.draw_information(env)



        pygame.display.update()


        self.clock.tick(10)


        return running



    def close(self):

        pygame.quit()