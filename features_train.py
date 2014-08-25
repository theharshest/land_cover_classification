import numpy as np
import scipy.fftpack
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import math

def get_features_fourier(dat):
	fourier = np.fft.fft(dat, axis=1)
	phase = np.angle(fourier)
	fourier = np.absolute(fourier)

	n = 3

	fourier = [ts[:n] for ts in fourier]
	
	return np.array(fourier)

def get_features_dct(dat):
	dat = dat.astype(float)
	dct = scipy.fftpack.dct(dat)

	n = 6 

	#print n

	dct = [np.hstack((ts[:n])) for ts in dct]
	return np.array(dct)

def get_features_dwt(dat):
	f = open('dwt_all.pickle', 'r')
	d = pickle.load(f)

	d = np.hstack((d[:,:5], d[:,16:21]))
	print d.shape

	return d

def apply_knn(train_set, test_set):
	'''
	Runs 1-NN on the dataset
	'''
	knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', n_neighbors=1, p=1, weights='uniform')
	knn.fit(train_set[:,:-1], train_set[:,-1])
	score = knn.score(test_set[:,:-1], test_set[:,-1])
	output = knn.predict(test_set[:,:-1])
	return score, output

def apply_svm(train_set, test_set):
	'''
	Runs Linear SVM classifier on the dataset
	'''
	clf = svm.LinearSVC()
	clf.fit(train_set[:,:-1], train_set[:,-1])
	
	score = clf.score(test_set[:,:-1], test_set[:,-1])
	output = clf.predict(test_set[:,:-1])
	return score, output

def prf(test_set, output):
	'''
	Returns Precision, Recall and F-measure
	'''
	labels = test_set[:,-1]
	tp = 0	
	fp = 0
	fn = 0

	for l,o in zip(labels, output):
		if l == o:
			if l == 2:
				tp += 1
		elif l == 1:
			if o == 2:
				fp += 1
		elif l == 2:
			if o == 1:
				fn += 1

	p = tp/float(tp+fp)
	r = tp/float(tp+fn)
	f = 2*p*r/(p+r)

	return p, r, f

if __name__ == '__main__':
	#Unpickling data
	f = open('train_data.pickle', 'r')
	data = pickle.load(f)

	data = data[1:]
	data = np.array(data, dtype=np.int32)

	m, n = data.shape

	# Shuffling data points so that while partioning the set for cross validation we get random points in each set
	#np.random.shuffle(data)

	# Change get_features_fourier to get_features_dct to change feature selection method from DFT to DCT
	data = np.hstack((get_features_fourier(data[:,:-1]), data[:,-1].reshape(m, 1)))

	print "Training k-NN.."
	knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', n_neighbors=1, p=1, weights='uniform')
	knn.fit(data[:,:-1], data[:,-1])
	f = open('train_knn.pickle', 'wb')
	pickle.dump(knn, f)
	f.close()
	raw_input('Training finished. Press enter to move to training SVM..')

	print "Training SVM.."
	clf = svm.LinearSVC()
	clf.fit(data[:,:-1], data[:,-1])
	f = open('train_linearsvm.pickle', 'wb')
	pickle.dump(clf, f)
	f.close()
	raw_input('Training finished.')

	# Uncomment below part to see the evaluation results on the train set

	'''
	offset = int(len(data)*0.2)
	x1 = 0
	x2 = offset

	al = []
	pl = []
	rl = []
	fl = []

	knn = 0 

	for i in range(5):
		train_set = np.vstack((data[:x1], data[x2:]))
		test_set = data[x1:x2]

		score, output = apply_knn(train_set, test_set)
		p, r, f = prf(test_set, output)

		al.append(score)
		pl.append(p)
		rl.append(r)
		fl.append(f)

		x1 += offset
		x2 += offset

	a = sum(al)/float(len(al))
	p = sum(pl)/float(len(pl))
	r = sum(rl)/float(len(rl))
	f = sum(fl)/float(len(fl))

	print "Accuracy =", a
	print "Precision =", p
	print "Recall =", r
	print "F-measure =", f
	'''
