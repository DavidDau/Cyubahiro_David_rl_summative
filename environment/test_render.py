from environment.rendering import NoiseRenderer
import time


renderer = NoiseRenderer()


for _ in range(100):

    renderer.draw_network(
        current_zone=0
    )

    time.sleep(0.05)


renderer.close()