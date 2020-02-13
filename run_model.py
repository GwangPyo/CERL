from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env

from navigation_env import NavigationEnvDefault, NavigationEnvAggressive, NavigationEnvDefensive, NavigationEnvPlanning, NavigationEnvMaster
from datetime import datetime
from logger.meta import get_model_dir, write_temp, get_key, GET_METADATA
from multiprocessing import Process, Lock

from subpolicy_loader import load_subpolicies

import copy

LoggerLock = Lock()

def logValue(RL_type, model_path, score, env_kwargs):
    ret = {}
    col = GET_METADATA()
    ret[col[1]] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ret[col[2]] = RL_type
    ret[col[3]] = score
    ret[col[4]] = model_path
    ret[col[5]] = env_kwargs["max_obs_range"]
    ret[col[6]] = env_kwargs["num_disturb"]
    ret[col[7]] = env_kwargs["tail_latency"]
    ret[col[8]] = env_kwargs["latency_accuracy"]
    ret[col[9]] = env_kwargs["initial_speed"]
    ret[col[10]] = env_kwargs["num_obstacle"] + env_kwargs["num_disturb"]
    ret[col[11]] = env_kwargs["max_delay"]
    return ret


def eval_model(env, model):
    scores = []
    while len(scores) < 1000:
        done = False
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                if reward > 0:
                    scores.append(1)
                else:
                    scores.append(0)
        print(len(scores))

    return sum(scores)/len(scores)


def RUN(env_type,  RL_type, env_kwargs, suffix=".zip", sample_number=None, log_tensorboard=True, n_envs=8, steps=10000000):
    path = get_model_dir(sample_number)
    model_dir = path + "/{}{}".format(RL_type, suffix)
    env = make_vec_env(lambda: env_type(**env_kwargs), n_envs=n_envs)
    if log_tensorboard:
        log_dir = path
    else:
        log_dir = None
    model = PPO2(env=env, policy=MlpPolicy, verbose=1, tensorboard_log=log_dir)
    model.learn(steps)
    model.save(model_dir)
    env = env_type(**env_kwargs)
    score = eval_model(env, model)
    logval = logValue(RL_type, model_dir, score, env_kwargs)
    write_temp(key=get_key(), log_values=logval)
    # to get model immediately
    return model_dir


def epoch(env_kwargs, suffix=".zip", sample_number=None, log_tensorboard=True):
    def template(n_envs=8, steps=10000000):
        return lambda env_type, RL_type: RUN(env_type=env_type, RL_type=RL_type, env_kwargs=env_kwargs,
            suffix=suffix, sample_number=sample_number, log_tensorboard=log_tensorboard, n_envs=n_envs, steps=steps)
    Template = template()
    default_model = Template(env_type=NavigationEnvDefault, RL_type="default")
    aggressive_model = Template(env_type=NavigationEnvAggressive, RL_type="aggressive")
    defensive_model = Template(env_type=NavigationEnvDefensive, RL_type="defensive")
    master_template = template(n_envs=4, steps=1000000)
    NavigationEnvMaster.set_subpolicies([default_model, aggressive_model, defensive_model])
    master_template(NavigationEnvMaster, RL_type="master")
    Template(NavigationEnvPlanning, RL_type="planning")


def eval_global(env_kwargs, suffix=".zip", sample_number=None, log_tensorboard=True):
        def template(n_envs=8, steps=10000000):
            return lambda env_type, RL_type: RUN(env_type=env_type, RL_type=RL_type, env_kwargs=env_kwargs,
                                                 suffix=suffix, sample_number=sample_number,
                                                 log_tensorboard=log_tensorboard, n_envs=n_envs, steps=steps)

        master_template = template(n_envs=4, steps=1000000)
        subpolicies = load_subpolicies(env_kwargs)
        NavigationEnvMaster.set_subpolicies(subpolicies)
        master_template(NavigationEnvMaster, RL_type="master")
        Template = template()
        Template(NavigationEnvPlanning, RL_type="planning")


default_env_kwargs = {
        "tail_latency":5,
        "max_obs_range":3,
        "num_disturb":4,
        "num_obstacle":4,
        "initial_speed":2,
        "latency_accuracy":0.95,
        "max_delay":2
    }


def eval_max_obs_range():
    max_obs_range = [3.5, 4]
    for s in max_obs_range:
        kwargs = copy.copy(default_env_kwargs)
        kwargs["max_obs_range"] = s
        epoch(env_kwargs=kwargs)


def eval_num_obstacles():
    num_obstacles = [6]
    for n in num_obstacles:
        kwargs = copy.copy(default_env_kwargs)
        kwargs["num_disturb"] = n
        kwargs["num_obstacle"] = n
        eval_global(kwargs)


def eval_num_disturbs():
    num_disturb = [0, 2]
    for n in num_disturb:
        kwargs = copy.copy(default_env_kwargs)
        kwargs["num_disturb"] = n
        epoch(env_kwargs=kwargs)



if __name__ == "__main__":
    eval_num_obstacles()