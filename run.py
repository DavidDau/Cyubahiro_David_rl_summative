from environment.custom_env import NoiseInspectionEnv


env = NoiseInspectionEnv()

obs, info = env.reset()

print("Starting state:")
print(obs)


done = False


while not done:

    action = env.action_space.sample()

    obs, reward, done, truncated, info = env.step(action)


    print("\nAction:", action)
    print("Reward:", reward)
    print("State:", obs)

    env.render()


print("Episode finished")