"""
Custom Gymnasium Environment:
Kigali Urban Noise Inspection Reinforcement Learning Environment
"""

import numpy as np
import gymnasium as gym

from gymnasium import spaces

from environment.map import RoadNetwork


class NoiseInspectionEnv(gym.Env):

    metadata = {
        "render_modes": ["human"]
    }


    def __init__(self, render_mode=None):

        super().__init__()

        self.render_mode = render_mode

        self.network = RoadNetwork()


        self.max_time = 80
        self.max_battery = 100


        # Actions:
        # 0 = Move
        # 1 = Move
        # 2 = Move
        # 3 = Move
        # 4 = Inspect
        # 5 = Wait

        self.action_space = spaces.Discrete(6)



        self.observation_space = spaces.Box(

            low=np.array(
                [0, 0, 0, 0, 0, 0, 0],
                dtype=np.float32
            ),

            high=np.array(
                [
                    10,
                    100,
                    50,
                    10,
                    10,
                    1,
                    1
                ],
                dtype=np.float32
            ),

            dtype=np.float32

        )


        self.current_zone = 0

        self.battery = self.max_battery

        self.remaining_time = self.max_time

        self.violations_found = 0


        self.renderer = None



    # -------------------------------------------------
    # Reset
    # -------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.network.reset()

        self.network.generate_noise_events()


        self.current_zone = 0

        self.battery = self.max_battery

        self.remaining_time = self.max_time

        self.violations_found = 0


        return self._get_observation(), {}



    # -------------------------------------------------
    # Step
    # -------------------------------------------------

    def step(self, action):

        reward = 0

        terminated = False

        truncated = False



        current = self.current_zone



        # -------------------------------
        # Movement
        # -------------------------------

        if action in [0, 1, 2, 3]:

            neighbours = self.network.get_neighbours(
                current
            )


            if neighbours:

                destination = self.np_random.choice(
                    neighbours
                )


                distance = self.network.get_distance(
                    current,
                    destination
                )


                self.current_zone = destination


                zone = self.network.get_zone(
                    destination
                )


                if not zone.visited:

                    zone.visited = True

                    reward += 10


                reward -= 1


                self.battery -= distance * 2


            else:

                reward -= 20



        # -------------------------------
        # Inspection
        # -------------------------------

        elif action == 4:


            zone = self.network.get_zone(
                self.current_zone
            )


            if zone.inspected:

                reward -= 50

                terminated = True

            else:

                zone.inspected = True

                if zone.has_violation:

                    zone.violation_detected = True

                    self.violations_found += 1

                    reward += 150


                else:

                    reward += 75



        # -------------------------------
        # Wait
        # -------------------------------

        elif action == 5:

            reward -= 30



        # Resource consumption

        self.battery -= 1

        self.remaining_time -= 1



        # -------------------------------
        # Mission completion
        # -------------------------------

        if self._mission_complete():

            reward += 300

            terminated = True



        elif self.remaining_time <= 0:

            reward -= 50

            terminated = True



        elif self.battery <= 0:

            reward -= 100

            terminated = True



        observation = self._get_observation()



        info = {

            "zone": self.current_zone,

            "violations": self.violations_found,

            "battery": self.battery,

            "time": self.remaining_time,

            "reward": reward

        }


        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )



    # -------------------------------------------------
    # Observation
    # -------------------------------------------------

    def _get_observation(self):

        zone = self.network.get_zone(
            self.current_zone
        )


        return np.array(

            [

                self.current_zone,

                self.battery,

                self.remaining_time,

                self.violations_found,

                self._number_inspected(),

                zone.risk_probability,

                1.0 if zone.inspected else 0.0

            ],

            dtype=np.float32

        )



    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _number_inspected(self):

        return sum(

            zone.violation_detected or zone.visited and zone.inspected

            for zone in self.network.zones.values()

        )



    def _mission_complete(self):

        return (

            self._number_inspected()

            >= 6

        )



    # -------------------------------------------------
    # Rendering
    # -------------------------------------------------

    def render(self):

        if self.renderer is None:

            from environment.rendering import NoiseRenderer

            self.renderer = NoiseRenderer()


        self.renderer.draw_network(
            self.current_zone
        )



    def close(self):

        if self.renderer:

            self.renderer.close()