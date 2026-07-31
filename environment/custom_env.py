"""
Custom Gymnasium Environment:
Kigali Urban Noise Inspection Reinforcement Learning Environment
"""

import numpy as np
import gymnasium as gym

from gymnasium import spaces

from environment.map import RoadNetwork


class NoiseInspectionEnv(gym.Env):
    """
    RL environment where an inspection vehicle navigates Kigali zones
    to detect noise pollution violations.
    """

    metadata = {
        "render_modes": ["human"]
    }


    def __init__(self, render_mode=None):

        super().__init__()

        self.render_mode = render_mode

        self.network = RoadNetwork()

        # Mission constraints
        self.max_time = 50
        self.max_battery = 100


        # Actions:
        # 0-9   : Move to zone
        # 10    : Inspect current zone
        # 11    : Wait

        self.action_space = spaces.Discrete(12)


        # Observation:
        # current_zone
        # battery
        # remaining_time
        # violations_found
        # inspected_zones
        # current_zone_risk

        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(6,),
            dtype=np.float32
        )


        self.current_zone = 0
        self.battery = self.max_battery
        self.remaining_time = self.max_time
        self.violations_found = 0


        self.renderer = None


    # -------------------------------------------------
    # Reset Environment
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



        # Move Action

        if action < 10:

            destination = action


            if destination in self.network.get_connections(
                self.current_zone
            ):


                distance = self.network.get_distance(
                    self.current_zone,
                    destination
                )


                self.current_zone = destination


                self.battery -= distance * 2

                self.remaining_time -= distance


                reward -= 2



            else:

                reward -= 50



        # Inspect Action

        elif action == 10:


            zone = self.network.get_zone(
                self.current_zone
            )


            if zone.inspected:

                reward -= 10


            else:

                zone.inspected = True


                if zone.has_violation:

                    reward += 100

                    self.violations_found += 1


                else:

                    reward += 20



        # Wait Action

        elif action == 11:


            reward -= 2

            self.remaining_time -= 1



        # Resource consumption

        self.battery -= 1

        self.remaining_time -= 1



        # Terminal conditions

        if self.battery <= 0:

            reward -= 100

            terminated = True



        elif self.remaining_time <= 0:

            reward -= 100

            terminated = True



        elif self._mission_complete():

            reward += 150

            terminated = True



        observation = self._get_observation()



        info = {

            "zone":
                self.current_zone,

            "violations":
                self.violations_found,

            "battery":
                self.battery,

            "time":
                self.remaining_time,

            "reward":
                reward

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

                zone.risk_probability

            ],

            dtype=np.float32
        )



    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _number_inspected(self):

        return sum(

            zone.inspected

            for zone in self.network.zones.values()

        )



    def _mission_complete(self):

        return (

            self._number_inspected()

            == self.network.number_of_zones()

        )



    # -------------------------------------------------
    # Rendering
    # -------------------------------------------------

    def render(self):

        if self.renderer is None:

            from environment.rendering import NoiseRenderer

            self.renderer = NoiseRenderer()


        return self.renderer.render(self)



    def close(self):

        if self.renderer:

            self.renderer.close()