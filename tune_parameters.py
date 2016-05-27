import itertools

def get_all_combinations(list1, list2):
	pairs = None
	if len(list1) > len(list2):
		permutations = itertools.permutations(list1,len(list2))
		pairs = [zip(perm, list2) for perm in permutations]
	else:
		permutations = itertools.permutations(list2, len(list1))
		pairs = [zip(list1, perm) for perm in permutations]
	combinations = []
	for p in pairs:
		for match in p:
			combinations.append(match[0] + match[1])
	return combinations

def tune_parameters(un_named_params, named_params, tuning_params):
'''
Method for tuning parameters during runtime. 

:param: un_named_params - a TUPLE containing parameters without default values
:param: named_params 	- a DICTIONARY mapping static parameter names to values
:param: tuning_params	- a DICTIONARY mapping names of tuning parameters to a 
							list specifying a range of values to tune
'''
keys = [key for key in tuning_params]
param_combos = [[elem] for elem in tuning_params[keys[0]]]
for i in range(1,len(keys)):
	key = keys[i]
	param_combos = get_all_combinations(param_combos, [[elem] for elem in tuning_params[key]])

for params in param_combos:
	param_dict = {}
	for i in range(len(keys)):
		param_dict[keys[i]] = params[i]
	
	portfolio = Portfolio(*un_named_params, **named_params, **param_dict)
	portfolio.run()

	# TODO: get sharpe ratio