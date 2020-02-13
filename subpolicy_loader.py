import sqlite3
from logger.meta import BINARY_DATA_PATH
from stable_baselines import PPO2


default_env_kwargs = {
        "tail_latency":5,
        "max_obs_range":3,
        "num_disturb":4,
        "num_obstacle":4,
        "initial_speed":2,
        "latency_accuracy":0.95,
        "max_delay":0
    }

SQL_QUERY = "SELECT model_path FROM ExpData where (rltype == \"{}\" and obs_range == {} and \
                    tail_latency == {} and \
                    init_speed == {} and \
                    disturb_num == {} and\
                    total_obstacle == {}) order by day desc; "

RL_TYPE = ["default", "aggressive", "defensive"]


def load_subpolicies(env_kwargs: dict) -> list:
    """
    :param env_kwargs:  get configuration of environment to search correct subpolicies
    :return: lists of subpolicy paths
    """
    db_path = BINARY_DATA_PATH + "/CERL.db"
    cursor = sqlite3.connect(db_path).cursor()
    paths= []
    for i in range(3):
        query = SQL_QUERY.format(RL_TYPE[i], env_kwargs["max_obs_range"],
                                                 env_kwargs["tail_latency"],
                                                 env_kwargs["initial_speed"],
                                                 env_kwargs["num_disturb"],
                                                 env_kwargs["num_obstacle"] + env_kwargs["num_disturb"])
        cursor.execute(query)
        model_path = cursor.fetchall()[0][0]
        paths.append(model_path)
    assert len(paths ) == 3, "some models are missing "

    return paths
