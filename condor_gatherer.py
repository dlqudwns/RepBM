
import numpy as np
num_repetition = 100
domain = 'cartpole'
factual_types = ['hard', 2.5, 2.0, 1.5, 1.0, 0.5]
methods = ['Baseline'] + ['mse_pi_{}'.format(ft) for ft in factual_types] +\
          ['repbm_{}'.format(ft) for ft in factual_types]
experiments = {method:([],[]) for method in methods}
for num in range(num_repetition):
    result_filepath = 'results/result_{}_{}.npy'.format(domain, num)
    try:
        result = np.load(result_filepath)
        print(result)
        for i, method in enumerate(methods):
            experiments[method][0].append(result[0, i])
            experiments[method][1].append(result[1, i])
    except:
        print('passed')

for key, value in experiments.items():
    print('({}) av_mse: {:.3f}±{:.3e}, ind_mse: {:.3f}±{:.3e}'.format(
        key, np.mean(value[0]).item(), (np.std(value[0])/np.sqrt(num_repetition)).item(),
        np.mean(value[1]).item(), (np.std(value[1])/np.sqrt(num_repetition)).item()))

'''
num_repetition = 100
domain = 'pendulum'
factual_types = ['hard', 2.5, 2.0, 1.5, 1.0, 0.5]
methods = ['Baseline'] + ['mse_pi_{}'.format(ft) for ft in factual_types] +\
          ['repbm_{}'.format(ft) for ft in factual_types]
experiments = {method:([],[]) for method in methods}
for num in range(num_repetition):
    result_filepath = 'results/6853.{}.out'.format(num)

    with open(result_filepath, 'r') as f:
        lines = f.readlines()
        result1 = eval(lines[-2])
        result2 = eval(lines[-1])
        for i, method in enumerate(methods):
            experiments[method][0].append(result1[i])
            experiments[method][1].append(result2[i])

for key, value in experiments.items():
    print('({}) av_mse: {:.3f}±{:.3e}, ind_mse: {:.3f}±{:.3e}'.format(
        key, np.mean(value[0]).item(), (np.std(value[0])/np.sqrt(num_repetition)).item(),
        np.mean(value[1]).item(), (np.std(value[1])/np.sqrt(num_repetition)).item()))
'''