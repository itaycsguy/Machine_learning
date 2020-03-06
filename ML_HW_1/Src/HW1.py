'''
.	Student Names:
.	==============
.	Itay Guy, I.D. - 305104184
.	Uri Ben-Izhak, I.D. - 066374737
'''

## Outter Packages
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

## Standard Packages
import pickle as pkl
import os


## Script Strict Information
APP_INFO = { 
	'dataset': '{}\\dataset', 
	'results': '{}\\results', 
	'train_indices': 'train_indices.txt', 
	'test_indices': 'test_indices.txt', 
	
	'train_size': 0.8, 
	'validation_size': 0.2, 
	
	'train_curve_title': 'Training Curve', 
	'validation_curve_title': 'Validation Curve', 
	'curves_title': 'Seeking Of The Optimal K', 
	'curves_name': 'Curves', 

	'train_curve_name': 'train_curve.pkl', 
	'validation_curve_name': 'validation_curve.pkl',  
	'optimal_k': 'optimal_k.pkl', 
	'validation_accuracy': 'validation_accuracy.pkl', 
	'test_accuracy': 'test_accuracy.pkl', 
	'train_mode': 0, 
	'test_mode': 1 
}

## Setup The Input Arguments From The User
def set_app_info(args):
	assert os.path.exists(args.root_dir), '{} does not exist.'.format(args.root_dir)
	APP_INFO['train_size'] = 1.0 - args.validation_size
	APP_INFO['validation_size'] = args.validation_size
	print('=== Setting Train-Set|Validation-Set [{}%:{}%] ==='.format(int(APP_INFO['train_size'] * 100), int(APP_INFO['validation_size'] * 100)))

	APP_INFO['results'] = APP_INFO['results'].format(args.root_dir)
	if not os.path.exists(APP_INFO['results']):
		print('=== Creating Directory: {} ==='.format(APP_INFO['results']))
		os.mkdir(APP_INFO['results'])
	else:
		print('=== Found: {} ==='.format(APP_INFO['results']))

	APP_INFO['dataset'] = APP_INFO['dataset'].format(args.root_dir)
	if args.dataset_dir is not None and os.path.exists(args.dataset_dir) and os.path.isdir(args.dataset_dir):
		print('=== Setting Directory: {} ==='.format(args.dataset_dir))
		APP_INFO['dataset'] = args.dataset_dir
	else:
		assert os.path.exists(APP_INFO['dataset']), '{} does not exist.'.format(APP_INFO['dataset'])
		print('=== Found: {} ==='.format(APP_INFO['dataset']))


def get_dataset(mndata, mode, seed):
	if mode in [APP_INFO['train_mode'], APP_INFO['test_mode']]:

		print('=== Loading Data ===')
		images, labels = mndata.load_training() if mode == APP_INFO['train_mode'] else mndata.load_testing()
		images, labels = reduce_dataset(images, labels, mode=mode)
		images, labels = np.asarray(images), np.asarray(labels)
		dataset = [images, labels]

	if mode == APP_INFO['train_mode']:

		print('=== Preparing Train-Set|Validation-Set [{}%:{}%] ==='.format(int(APP_INFO['train_size'] * 100), int(APP_INFO['validation_size'] * 100)))
		np.random.seed(seed)

		## separating the data to train/validation sets
		total_size = len(images)

		permut = np.random.permutation(total_size)
		permut_images, permut_labels = list(), list()
		for p in permut:
			permut_images.append(images[p])
			permut_labels.append(labels[p])

		trainset_images = permut_images[:int(total_size * APP_INFO['train_size'])]
		trainset_labels = permut_labels[:int(total_size * APP_INFO['train_size'])]

		validationset_images = permut_images[int(total_size * APP_INFO['train_size']):]
		validationset_labels = permut_labels[int(total_size * APP_INFO['train_size']):]

		dataset = [trainset_images, trainset_labels, validationset_images, validationset_labels]

	return dataset



## Executing Single Iteration With Specific K
def test_k(image_set, label_set, k):
	from operator import itemgetter
	acc_list = list()
	## fitting the k
	for idx, (img, lbl) in enumerate(zip(image_set, label_set)):
		dists = list()
		for nidx, (nimg, nlbl) in enumerate(zip(image_set, label_set)):
			if idx != nidx:
				## this is not me
				dists.append([np.linalg.norm(img - nimg, 2), nlbl])
			
		sorted_kdists = sorted(dists, key=itemgetter(0))[:k]
		dists_labels = [sorted_kdists[i][1] for i in range(len(sorted_kdists))]
		max_freq_label = np.argmax(np.bincount(dists_labels))
		acc_list.append(max_freq_label == lbl)

	fit_acc = sum(acc_list) / len(acc_list)
	return fit_acc


## Fitting The Data For All K
def fit(image_set, label_set):
	from operator import itemgetter
	perf_curve = list()
	max_k = int(np.ceil(np.sqrt(len(image_set) + 1))) ## a common heuristic approach
	for k in range(1, max_k):
		perf_curve.append((k, test_k(image_set, label_set, k)))

	return perf_curve


## Heuristic Procedure To Find Some Approximation Of The Optimal K Without Ant-Snapping
def find_best_k(train_perf_curve, validation_perf_curve, smooth=True, sigma=.5):
	def min_loss_idx(mi_len):
		L = list()
		alpha, beta = .7, .3
		# loss(pt, pv) = alpha * (pt - pv)**2 - beta * (npt.dot(npv)) -> min
		for i in range(mi_len):
			tprev_point = 0.0
			vprev_point = 0.0

			tnext_point = 0.0
			vnext_point = 0.0

			if i > 0:
				tprev_point = train_perf_curve[i - 1][1]
				vprev_point = validation_perf_curve[i - 1][1]

			if i < (mi_len - 1):
				tnext_point = train_perf_curve[i + 1][1]
				vnext_point = train_perf_curve[i + 1][1]

			tv_dist = (train_perf_curve[i][1] - validation_perf_curve[i][1])**2
			tprev_vec = (train_perf_curve[i][1], tprev_point - train_perf_curve[i][1])
			tnext_vec = (train_perf_curve[i][1], tnext_point - train_perf_curve[i][1])
			tnormal = np.cross(tprev_vec, tnext_vec)

			vprev_vec = (validation_perf_curve[i][1], vprev_point - validation_perf_curve[i][1])
			vnext_vec = (validation_perf_curve[i][1], vnext_point - validation_perf_curve[i][1])
			vnormal = np.cross(vprev_vec, vnext_vec)
			
			loss = alpha * tv_dist - beta * tnormal.dot(vnormal)
			L.append(loss)

		# L = [loss(0), ..., loss(min(|train_perf_curve|, |validation_perf_curve|))]
		return np.argmin(L)


	if smooth:
		tc = gaussian_filter([proba[1] for proba in train_perf_curve], sigma=sigma)
		for i in range(len(tc)):
			train_perf_curve[i] = (train_perf_curve[i][0], tc[i])

		vc = gaussian_filter([proba[1] for proba in validation_perf_curve], sigma=sigma)
		for i in range(len(vc)):
			validation_perf_curve[i] = (validation_perf_curve[i][0], vc[i])

	mi_len = min(len(train_perf_curve), len(validation_perf_curve))
	return train_perf_curve[min_loss_idx(mi_len)][0]


## Plotting Out The Results To A Graph Included The Approximation Optimal K
def plot_curves(fit_perf_curves, log_scale=False, smooth=False, sigma=.5, k=None):
	def set_xlims(lims, mi, ma):
		if lims['max'] == 0:
			lims['max'] = ma
		else:
			lims['max'] = min(lims['max'], ma)

		if lims['min'] > mi:
			lims['min'] = mi
		return lims

	def set_ylims(lims, mi, ma):
		if lims['max'] < ma:
			lims['max'] = ma

		if lims['min'] > mi:
			lims['min'] = mi
		return lims


	fig, ax = plt.subplots(len(fit_perf_curves.items()))
	plt.subplots_adjust(hspace=0.35, wspace=0)
	fig.suptitle('{} {}'.format(APP_INFO['curves_title'], '(Log2-Scale)' if log_scale else ''))
	xlims = {'min': 0.0, 'max': 0.0}
	ylims = {'min': 0.0, 'max': 1.0}
	for i, (curves_name, curve) in enumerate(fit_perf_curves.items()):
		Ks = [k[0] for k in curve]
		Probas = [proba[1] for proba in curve]
		
		xlims = set_xlims(xlims, min(Ks), max(Ks))
		if log_scale:
			np.seterr(divide='ignore')
			Probas = np.log2(Probas)
			np.seterr(divide='warn')
			Probas[Probas == np.inf] = np.nan
			Probas[Probas == -np.inf] = np.nan
			ylims = set_ylims(ylims, min(Probas), max(Probas))

		if smooth:
			Probas = gaussian_filter(Probas, sigma=sigma)
		
		ax[i].plot(Ks, Probas)
		ax[i].plot(Ks[k], Probas[k], 'g*', color='red')
		ax[i].set_title(curves_name)

	for i in range(len(fit_perf_curves.items())):
		ax[i].set_xlim(left=xlims['min'], right=xlims['max'])
		ax[i].set_ylim(bottom=ylims['min'], top=ylims['max'])

	return plt.gcf()


## Loading The Data From `results` Directory By Name - Required For The Script To Run After Training Time
def load(name):
	result = None
	path = '{}\\{}'.format(APP_INFO['results'], name)
	with open(path, 'rb') as f:
		result = pkl.load(f)

	return result


## Saving All The Required Data To `results` Directory
def save(name, result):
	path = '{}\\{}'.format(APP_INFO['results'], name)
	with open(path, 'wb+') as f:
		pkl.dump(result, f)


def reduce_dataset(images, labels, mode):
	print('=== Mapping Indices From Dataset ===')
	indices = list()
	if mode == APP_INFO['train_mode']:
		train_indices_path = '{}\\{}'.format(APP_INFO['dataset'], APP_INFO['train_indices'])
		with open(train_indices_path, 'r') as tf:
			for line in tf:
				indices.append(int(line.strip().replace(" ", "")))

	elif mode == APP_INFO['test_mode']:
		test_indices_path = '{}\\{}'.format(APP_INFO['dataset'], APP_INFO['test_indices'])
		with open(test_indices_path, 'r') as tf:
			for line in tf:
				indices.append(int(line.strip().replace(" ", "")))

	print('=== Found: {} Indices ==='.format(len(indices)))
	reduced_images, reduced_labels = list(), list()
	for idx in indices:
		reduced_images.append(images[idx])
		reduced_labels.append(labels[idx])

	return reduced_images, reduced_labels



def get_best_k(seek_k, train_perf_curve, validation_perf_curve):
	if not seek_k:
		print('=== Loading The Optimal K ===')
		opt_k = load(APP_INFO['optimal_k'])
	else:
		print('=== Seeking The Optimal K ===')
		opt_k = find_best_k(train_perf_curve, validation_perf_curve)
		save(APP_INFO['optimal_k'], opt_k)

	return opt_k


## Script Main
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="KNN Algorithm On MNIST Dataset.")
	parser.add_argument('-root_dir', type=str, default='.\\')
	parser.add_argument('-dataset_dir', type=str, default=None)
	parser.add_argument('-seek_k', action='store_true', default=False)
	parser.add_argument('-train', action='store_true', default=False)
	parser.add_argument('-validation_size', type=float, default=0.2)
	parser.add_argument('-test', action='store_true', default=False)
	parser.add_argument('-plot_curves', action='store_true', default=False)
	parser.add_argument('-plot_smooth_curves', action='store_true', default=False)
	parser.add_argument('-plot_log_scale', action='store_true', default=False)
	parser.add_argument('-seed', type=int, default=42)
	args = parser.parse_args()
	print('=== Setup Environment ===')
	set_app_info(args)

	## Loading the data by mnist package to simplify the process
	mndata = MNIST(APP_INFO['dataset'])
	figure = None

 	## train
	if args.train:
		print('=== Executing Training ===')
		trainset_images, trainset_labels, validationset_images, validationset_labels = get_dataset(mndata, mode=APP_INFO['train_mode'], seed=args.seed)

		print('=== Train Fitting ===')
		train_perf_curve = fit(trainset_images, trainset_labels)
		
		print('=== Validation Fitting ===')
		validation_perf_curve = fit(validationset_images, validationset_labels)

		print('=== Saving Curves ===')
		save(APP_INFO['train_curve_name'], train_perf_curve)
		save(APP_INFO['validation_curve_name'], validation_perf_curve)
		curves = {APP_INFO['train_curve_title']: train_perf_curve, APP_INFO['validation_curve_title']: validation_perf_curve}

		print('=== Seeking The Optimal K ===')
		opt_k = find_best_k(train_perf_curve, validation_perf_curve)
		save(APP_INFO['optimal_k'], opt_k)

		print('=== Executing Validation With K={} ==='.format(opt_k))
		validation_acc = test_k(validationset_images, validationset_labels, k=opt_k)
		save(APP_INFO['validation_accuracy'], validation_acc)

		print('=== Saving Curves As PNG Format ===')
		figure = plot_curves(curves, log_scale=args.plot_log_scale, smooth=args.plot_smooth_curves, k=opt_k)
		figure.savefig('{}\\{}'.format(APP_INFO['results'], APP_INFO['curves_name']))

		print('=== Done ===')

	else:
		print('=== Loading Curves ===')
		train_perf_curve, validation_perf_curve = load(APP_INFO['train_curve_name']), load(APP_INFO['validation_curve_name'])
		curves = {APP_INFO['train_curve_title']: train_perf_curve, APP_INFO['validation_curve_title']: validation_perf_curve}

	opt_k = get_best_k(args.seek_k, train_perf_curve, validation_perf_curve)
	if args.plot_curves:
		print('=== Plotting Curves ===')
		if figure is None:
			figure = plot_curves(curves, log_scale=args.plot_log_scale, smooth=args.plot_smooth_curves, k=opt_k)
			if args.seek_k:
				print('=== Saving Curves As PNG Format ===')
				figure.savefig('{}\\{}'.format(APP_INFO['results'], APP_INFO['curves_name']))
		plt.show()

	validation_acc = load(APP_INFO['validation_accuracy'])
	print('=== Validation Accuracy: {} ==='.format(validation_acc))
	print('=== Done ===')

	# test
	if args.test or args.seek_k:
		print('=== Executing Testing ===')

		test_images, test_labels = get_dataset(mndata, mode=APP_INFO['test_mode'], seed=args.seed)
		opt_k = load(APP_INFO['optimal_k'])
		test_accuracy = test_k(test_images, test_labels, k=opt_k)
		save(APP_INFO['test_accuracy'], test_accuracy)

	else:
		opt_k = load(APP_INFO['optimal_k'])
		test_accuracy = load(APP_INFO['test_accuracy'])

	print('=== Executing Test With K={} ==='.format(opt_k))
	print('=== Test Accuracy: {} ==='.format(test_accuracy))
	print('=== Done ===')