import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)
register(
    id='Navigation-v0',
    entry_point='navigation_env:NavigationEnvDefault',
    max_episode_steps=10000
)

register(
    id='Navigation-v1',
    entry_point='navigation_env:NavigationEnvAggressive',
    max_episode_steps=10000,
)

register(
    id='Navigation-v2',
    entry_point='navigation_env:NavigationEnvDepensive',
    max_episode_steps=10000,
)