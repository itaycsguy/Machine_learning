'''
.	Student Names:
.	==============
.	Itay Guy, I.D. - 305104184
.	Uri Ben-Izhak, I.D. - 066374737
'''
import pandas as pd
import numpy as np


## Linear Regression Task Methods
'''
.	Loading regression data.
.	Return datax, datay [lists]
'''
def load_data(input_file):
	FLIP_SIGN_AT = 0
	datax, datay = list(), list()
	with open(input_file, 'r') as file:
		for line in file:
			line = line.split(" ") ## known from the data
			new_line = list()
			for i in range(len(line)):
				if len(line[i]) > 0: ## avoiding white letters
					line[i] = np.float32(line[i].strip().replace(" ", ""))
					new_line.append(line[i])

			datax.append(new_line[:(len(new_line) - 1)])
			datay.append(new_line[(len(new_line) - 1)])
	
	return datax, datay


'''
.	Spliting the data to train & test chunks.
. Return xtrain, xtest, ytrain, ytest [numpy arrays]
'''
def train_test_split(X, y, train_size, random_state):
	length = len(X)
	np.random.seed(random_state)
	perm = np.random.permutation(length)
	X_new, y_new = list(), list()
	for i in range(length):
		X_new.append(X[perm[i]])
		y_new.append(y[perm[i]])

	xtrain = np.asarray(X_new[:train_size])
	ytrain = np.asarray(y_new[:train_size])
	xtest = np.asarray(X_new[train_size:])
	ytest = np.asarray(y_new[train_size:])
	return xtrain, xtest, ytrain, ytest


'''
.	Computing regression errors.
.	Return MSE, MAE
'''
def compute_reg_errors(ytrain_true, ytrain_pred, ytest_true, ytest_pred):
	mse_train = np.square(ytrain_true - ytrain_pred).mean(axis=0)
	mse_test = np.square(ytest_true - ytest_pred).mean(axis=0)
	return mse_train, mse_test


'''
.	Computing regrssion model optimal weights.
.	Return a [weights vector]
'''
def compute_reg_weigths(Y, b):
	YtY =  np.matmul(Y.T, Y)
	a = np.matmul(np.matmul(np.linalg.inv(YtY), Y.T), b)
	return a


'''
.	Computing predictions using MSE model.
.	Return numpy array predictions
'''
def compute_reg_predict(Y, a):
	return np.matmul(Y, a)
	

'''
.	Executing Linear Regression algorithm.
'''
def exec_REG(args):
	print('=== Executing Regrssion Task ===')
	datax, datay = load_data(args.input_file)
	datax, datay = np.asarray(datax), np.asarray(datay)
	datax = np.c_[np.ones((len(datax), 1)), datax] ## make homogenoues coordinate for MSE linear regression
	for train_size in args.train_sizes:
		test_size = len(datax) - train_size
		print('=== Train-Size/Test-Size: {}/{} >> '.format(train_size, test_size))
		Y_train, Y_test, b_train, b_test = train_test_split(datax, datay, train_size=train_size, random_state=args.random_state)
		
		a = compute_reg_weigths(Y_train, b_train)
		b_train_estimate = compute_reg_predict(Y_train, a)
		b_test_estimate = compute_reg_predict(Y_test, a)
	
		if args.plot_reg_steps:
			import matplotlib.pyplot as plt
			plt.scatter(b_train, b_train_estimate, c='red')
			plt.xlabel('Price')
			plt.ylabel('Predicted Price') 
			plt.title('Training {}-Samples'.format(train_size))
			plt.show()

			plt.scatter(b_test, b_test_estimate, c='green')
			plt.xlabel('Price')
			plt.ylabel('Predicted Price') 
			plt.title('Testing {}-Samples'.format(test_size))
			plt.show()

		mse_train, mse_test = compute_reg_errors(b_train, b_train_estimate, b_test, b_test_estimate)
		print(' >> Train-MSE/Test-MSE: {}/{}'.format(mse_train, mse_test))

	print('=== Done ===')



## LDA Task Methods
'''
.	Computing LDA Projections.
.	Return projections, projection matrix [numpy arrays]
'''
def lda(class1, class2, class3, dim):
	def _compute_scatter_within(classes):
		total_Sw = None
		for classk in classes:
			mean = np.mean(classk, keepdims=True, axis=0).tolist()[0]
			class_Sw = None
			for x in classk:
				x = np.reshape(x, (x.shape[0], 1))
				math_term = np.matmul(x - mean, x - mean)
				class_Sw = math_term if class_Sw is None else class_Sw + math_term

			total_Sw = class_Sw if total_Sw is None else total_Sw + class_Sw

		return total_Sw

			
	def _compute_scatter_between(classes):
		mean = np.asarray(np.mean(np.concatenate(tuple(classes), axis=0), keepdims=True, axis=0).tolist()[0])
		mean = np.reshape(mean, (mean.shape[0], 1))

		total_Sb = None
		for classk in classes:
			nk = classk.shape[0]
			meank = np.asarray(np.mean(classk, keepdims=True, axis=0).tolist()[0])
			meank = np.reshape(meank, (meank.shape[0], 1))
			math_term = nk * np.matmul(meank - mean, (meank - mean).T)
			total_Sb = math_term if total_Sb is None else total_Sb + math_term

		return total_Sb

	## Function Entry-Point
	Sw = _compute_scatter_within([class1, class2, class3])
	Sb = _compute_scatter_between([class1, class2, class3])

	from scipy.linalg import eig
	w, v = eig(Sw, Sb)
	ev_indices = np.argsort(w)[::-1][:dim].tolist()
	V = v[ev_indices]
	data = np.concatenate(tuple([class1, class2, class3]), axis=0).T
	Y = np.matmul(V, data)
	return Y, V


'''
.	Computing Leave-One-Out-Cross-Validation using LDA Modeling.
.	Return confusion_matrix, accuracy, covariance type matrix
'''
def LOOCV_3x3_by_LDA(Y, b, dim):
	## Computing the gaussian distribution
	def _compute_dist(data):
		c_mean = np.asarray(np.mean(data, keepdims=True, axis=0).tolist()[0])
		c_cov = None
		data_probs = list()
		for x in data:
			x = np.reshape(x, (1, x.shape[0]))
			math_term = np.matmul((x - c_mean).T, x - c_mean)
			c_cov = math_term if c_cov is None else c_cov + math_term

		return c_mean, (1 / data.shape[0]) * c_cov

	## Computing the decision function by covariance matrix case
	def _compute_class_desicion(x, c_mean, c_cov, prior=0.0, cov_case='diag'):
		if cov_case is 'diag':
			c_cov = np.diag(np.diag(c_cov))
			pow_sigma = np.linalg.det(c_cov)**2
			meut_div_pow_sigma_x = np.matmul(c_mean.T / pow_sigma, x)
			neg_meut_l2_div_2_pow_sigma = -np.matmul(c_mean.T, c_mean) / (2 * pow_sigma)
			ln_p = np.log2(prior)
			gx = (meut_div_pow_sigma_x + neg_meut_l2_div_2_pow_sigma + ln_p)[0]

		elif cov_case is 'same':
			cov_inv = np.linalg.inv(c_cov)
			ln_p = np.log2(prior)
			meut_sigma_inv_meu = np.matmul(np.matmul(c_mean.T, cov_inv), c_mean)
			meut_sigma_inv_x = np.matmul(np.matmul(c_mean.T, cov_inv), x)
			gx = (meut_sigma_inv_x + ln_p - (1 / 2) * meut_sigma_inv_meu)[0]

		elif cov_case is 'arbitrary':
			xt_sigma_inv_x = np.matmul(np.matmul(x.T, -(1 / 2) * np.linalg.inv(c_cov)), x)
			meut_sigma_inv_x = -(1 / 2) * np.matmul(np.matmul(c_mean.T, -(1 / 2) * np.linalg.inv(c_cov)), x)
			ln_cov_det = -(1 / 2) * np.log(np.linalg.det(c_cov))
			ln_p = np.log2(prior)
			gx = (xt_sigma_inv_x + meut_sigma_inv_x - (1 / 2) * meut_sigma_inv_x + ln_cov_det + ln_p)[0][0]

		return gx


	## Function Entry-Point
	confusion_matrices = list()
	accuracies = list()
	## Testing all kind of covariance matrices to pick to largest accuracy
	cov_cases, Y_CLASS_1, Y_CLASS_2, Y_CLASS_3, Y_DIM = ['diag', 'same', 'arbitrary'], 1, 2, 3, 3
	for cov_case in cov_cases:
		confusion_matrix = np.zeros((Y_DIM, Y_DIM))
		for k in range(Y.shape[0]):
			bk = np.delete(b, k, 0)
			Yk = np.delete(Y, k, 0)

			gap = Yk.shape[0] // Y_DIM
			c1_mean, c1_cov = _compute_dist(Yk[:(Y_CLASS_1 * gap)])
			c2_mean, c2_cov = _compute_dist(Yk[(Y_CLASS_1 * gap):(Y_CLASS_2 * gap)])
			c3_mean, c3_cov = _compute_dist(Yk[(Y_CLASS_2 * gap):])

			## desicion machines (single cluster regions)
			x = np.reshape(Y[k], (Y[k].shape[0], 1))
			cov1, cov2, cov3 = c1_cov, c2_cov, c3_cov
			if cov_case in ['diag', 'same']:
				cov1, cov2, cov3 = c3_cov, c3_cov, c3_cov

			d1 = _compute_class_desicion(x, c1_mean, cov1, prior=(1 / Y_DIM), cov_case=cov_case)
			d2 = _compute_class_desicion(x, c2_mean, cov2, prior=(1 / Y_DIM), cov_case=cov_case)
			d3 = _compute_class_desicion(x, c3_mean, cov3, prior=(1 / Y_DIM), cov_case=cov_case)

			est_c = np.argmax([d1, d2, d3])
			confusion_matrix[int(b[k]) - 1, est_c] += 1

		confusion_matrices.append(confusion_matrix)
		accuracies.append(confusion_matrix.trace() / Y.shape[0])

	accuracy_idx, accuracy = np.argmax(accuracies), np.max(accuracies)
	confusion_matrix = confusion_matrices[accuracy_idx]
	return confusion_matrix, accuracy, cov_cases[accuracy_idx]


'''
.	Computing Leave-One-Out-Cross-Validation using MSE Modeling for multiple classes.
.	Return confusion matrix, accuracy
'''
def LOOCV_3x3_by_MSE(Y, B):
	Y_DIM = 3
	confusion_matrix = np.zeros((Y_DIM, Y_DIM))
	Y = np.c_[np.ones((len(Y), 1)), Y]
	for k in range(Y.shape[0]):
		Bk = np.delete(B, k, 0)
		Yk = np.delete(Y, k, 0)
		
		Bk_pred = compute_reg_predict(Y[k], compute_reg_weigths(Yk, Bk))
		pred_class = np.argmax(Bk_pred)
		true_class = int(np.argmax(B[k]))
		confusion_matrix[true_class, pred_class] += 1

	return confusion_matrix, (confusion_matrix.trace() / Y.shape[0])


'''
.	Executing LDA algorithm.
'''
def exec_LDA(args):
	def _get_indicator_vec(idx, length):
		indicator = np.zeros(length)
		indicator[idx] = 1
		return indicator.tolist()

	print('=== Executing LDA Task ===')
	from scipy.io import loadmat
	db = loadmat(args.input_mat)
	classes = [db['class1'], db['class2'], db['class3']]
	Y_CLASS_1, Y_CLASS_2, Y_CLASS_3, Y_DIM = 1, 2, 3, 3

	colors = list()	
	print(' >> Class 1 tuned as red color.')
	print(' >> Class 2 tuned as green color.')
	print(' >> Class 3 tuned as blud color.')
	for i, classi in enumerate(classes):
		for _ in classi:
			if i is (Y_CLASS_1 - 1):
				colors.append('red')
			elif i is (Y_CLASS_2 - 1):
				colors.append('green')
			elif i is (Y_CLASS_3 - 1):
				colors.append('blue')

	for dim in args.reduction_dims:
		[Y, _] = lda(*classes, dim)

		## Display Scatter Plots
		import matplotlib.pyplot as plt
		plt.scatter(Y[0], np.zeros(Y[0].shape[0]) if dim is 1 else Y[1], color=colors, alpha=0.75)
		plt.title('Dimension={}'.format(dim))

		print('=== LDA Confusion Matrix By LOOCV With Dimension={} >> '.format(dim))
		gap = Y.shape[1] // Y_DIM
		b = np.concatenate(tuple([Y_CLASS_1 * np.ones(gap), Y_CLASS_2 * np.ones(gap), Y_CLASS_3 * np.ones(gap)]), axis=0)
		confusion_matrix, accuracy, cov_case = LOOCV_3x3_by_LDA(Y.T, b, dim)
		print(confusion_matrix)
		print(' >> LDA {}-Dimensional Success-Rate {} using {} covariance matrix'.format(dim, accuracy, cov_case))
		plt.show(block=True)
		
	## MSE Classification for all dimensionals
	print('=== MSE Confusion Matrix By LOOCV >> ')
	gap = Y.shape[1] // Y_DIM
	B = np.zeros((Y.shape[1], Y_DIM))
	for i in range(Y.shape[1]):
		if i < gap:
			B[i] = _get_indicator_vec(Y_CLASS_1 - 1, Y_DIM)
		elif i >= gap and i < (Y_CLASS_2 * gap):
			B[i] = _get_indicator_vec(Y_CLASS_2 - 1, Y_DIM)
		elif i >= (Y_CLASS_2 * gap) and i < (Y_CLASS_3 * gap):
			B[i] = _get_indicator_vec(Y_CLASS_3 - 1, Y_DIM)
	
	Y = np.concatenate(tuple([*classes]), axis=0)
	confusion_matrix, accuracy = LOOCV_3x3_by_MSE(Y, B)
	print(confusion_matrix)
	print(' >> MSE Success-Rate: {}'.format(accuracy))
	print('=== Done ===')



if __name__ == "__main__":
	import argparse, os
	parser = argparse.ArgumentParser(description="LDF Topics.")
	parser.add_argument('-root', type=str, default='..', help='App assumes you are executing it from the source code directory.')
	parser.add_argument('-input_file', type=str, default='../Dataset/housing.data', help='Linear regression data file.')
	parser.add_argument('-input_mat', type=str, default='../Dataset/P3.mat', help='LDA data matrix.')
	parser.add_argument('-train_sizes', type=list, default=[10, 50, 100, 200, 300, 400], help='App assumes the default list as input.')
	parser.add_argument('-random_state', type=int, default=42, help='App assumes the default randomize seed.')
	parser.add_argument('-use_reg', action='store_true', default=True, help='Enable/Disable linear regression task functionality.')
	parser.add_argument('-plot_reg_steps', action='store_true', default=False, help='Enable/Disable linear regression task plot regression steps.')
	parser.add_argument('-use_lda', action='store_true', default=True, help='Enable/Disable LDA task functionality.')
	parser.add_argument('-reduction_dims', type=list, default=[1, 2], help='App assumes the default list as input.')
	args = parser.parse_args()
	assert os.path.exists(args.root), 'root project directory does not exist.'

	## Regression Main Task
	if args.use_reg:
		assert os.path.exists(args.input_file), 'input_file does not exist.'
		assert args.train_sizes == [10, 50, 100, 200, 300, 400], 'Incorrect Train Sizes Array.'

		exec_REG(args)


	## LDA Main Task
	if args.use_lda:
		assert os.path.exists(args.input_mat), 'input_mat does not exist.'
		assert np.min(args.reduction_dims) >= 1 and np.max(args.reduction_dims) <= 2, 'Dimensionality Redutions should being 1 or 2 dimensions only.'

		if args.use_reg:
			print('')
		
		exec_LDA(args)