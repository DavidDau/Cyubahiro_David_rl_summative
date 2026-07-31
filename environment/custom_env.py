"""
Custom Gymnasium environment for Kigali Urban Noise Inspection.

The RL agent represents an environmental inspection vehicle.
Its goal is to discover noise regulation violations while
minimizing travel cost and operational resource usage.
"""

import random

import gymnasium as gym
import numpy as np

from gymnasium import spaces

from environment.map import RoadNetwork


class NoiseInspectionEnv(gym.Env):
    """
    Reinforcement learning environment for urban noise inspection.
    """

    metadata = {
        "render_modes": ["human"]
    }


    def __init__(self):

        super().__init__()


        # -----------------------------
        # Environment
        # -----------------------------

        self.network = RoadNetwork()


        # Mission constraints

        self.max_time = 50

        self.max_battery = 100



        # -----------------------------
        # Action Space
        #
        # 0-9   -> Move to zone
        # 10    -> Inspect current zone
        # 11    -> Wait
        # -----------------------------

        self.action_space = spaces.Discrete(12)



        # -----------------------------
        # Observation Space
        #
        # 0 Current zone
        # 1 Battery
        # 2 Remaining time
        # 3 Violations found
        # 4 Inspected zones
        # 5 Current zone risk
        # -----------------------------

        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(6,),
            dtype=np.float32
        )


        self.reset()



    # ==================================================
    # RESET
    # ==================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)


        self.network.reset()


        # Generate hidden violations

        self.network.generate_noise_events()



        # Agent starts at inspection base

        self.current_zone = 0


        self.remaining_time = self.max_time


        self.battery = self.max_battery


        self.violations_found = 0


        self.total_reward = 0



        observation = self._get_observation()


        return observation, {}



    # ==================================================
    # STEP
    # ==================================================

    def step(self, action):

        reward = 0


        terminated = False



        # --------------------------------------
        # MOVE ACTION
        # --------------------------------------

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



                reward -= distance



            else:

                # Invalid road

                reward -= 25



        # --------------------------------------
        # INSPECT ACTION
        # --------------------------------------

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



        # --------------------------------------
        # WAIT ACTION
        # --------------------------------------

        elif action == 11:

            reward -= 2

            self.remaining_time -= 1



        # --------------------------------------
        # Mission Update
        # --------------------------------------

        self.total_reward += reward



        # Battery drain

        self.battery -= 1


        self.remaining_time -= 1



        # --------------------------------------
        # Termination
        # --------------------------------------

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

            "current_zone":
                self.current_zone,

            "violations":
                self.violations_found,

            "battery":
                self.battery,

            "time":
                self.remaining_time,

            "reward":
                self.total_reward

        }



        return (
            observation,
            reward,
            terminated,
            False,
            info
        )



    # ==================================================
    # HELPERS
    # ==================================================

    def _mission_complete(self):

        inspected = 0


        for zone in self.network.zones.values():

            if zone.inspected:

                inspected += 1


        return inspected == self.network.number_of_zones()



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



    def _number_inspected(self):

        count = 0


        for zone in self.network.zones.values():

            if zone.inspected:

                count += 1


        return count



    # ==================================================
    # Visualization Placeholder
    # ==================================================

    def render(self):

        zone = self.network.get_zone(
            self.current_zone
        )


        print("--------------------------------")

        print(
            "Current Zone:",
            zone.name
        )

        print(
            "Battery:",
            self.battery
        )

        print(
            "Time Remaining:",
            self.remaining_time
        )

        print(
            "Violations Found:",
            self.violations_found
        )

        print(
            "Reward:",
            self.total_reward
        )

        print("--------------------------------")