import argparse
import os
import sys
import subprocess

# Define the project root and append it to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

exp_dict = {
    'f1' : 'exp_f1',
    'size': 'exp_classifiers_sizes',
    'cf_dist': 'exp_counterfactual_distance_reduction',
    'cf_size': 'exp_counterfactual_l0_norm',
    'param_search': 'exp_hyperparameter_search',
    'time': 'exp_time_effeciency'
}

def run_experiment(experiment_name):
    experiment_script = f'experiments/{exp_dict[experiment_name]}.py'
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    env['PROJECT_ROOT'] = project_root
    subprocess.run(['python', experiment_script], env=env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('experiment', type=str, help='''
                        Name of the experiment as one of the following (param_search, size, time, cf_dist, cf_size):\n
                        1- f1: the f1 score of each tested model on each dataset.\n
                        2- param_search: hyperparameter search of all the tested models.\n
                        3- size: the size of the classifiers to test local interpretability.\n
                        4- time: the training-prediction time in addition to the time of explanation components.\n
                        5- cf_dist: the L1 distance reduction of counterfactuals of E-IPS-KNN over IPS-KNN.\n
                        6- cf_size: the size reduction of counterfactual explanations of E-IPS-KNN over IPS-KNN.\n
                        ''')
    
    args = parser.parse_args()
    run_experiment(args.experiment)
