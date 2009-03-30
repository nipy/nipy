# This file is temporary work, just not use it as it's isn't quite tested, and not integrated. This wasn't even intended to be
# commited on the repo in the first place... Don't even read it unless you know you want to :-)
# - MiniHRF (a basic class to compute a canonical glover function and derivative)
# - bad example code which subsample a fake eventlist to generate a kind of designmatrix

# you can run this file as a standalone script

import numpy as np

# This class is basically the nipy glover computation code without the associated dependancy hell and bazillion classes
class MiniHRF:
	def __init__(self, peak_hrf=(5.4, 10.8), fwhm_hrf=(5.2, 7.35), dip=0.35, maxderiv = 3):
		def glover2GammaDENS(peak_hrf, fwhm_hrf, c = 1):
			alpha = np.power(peak_hrf / fwhm_hrf, 2) * 8 * np.log(2.0)
			beta = np.power(fwhm_hrf, 2) / peak_hrf / 8 / np.log(2.0)
			coef = peak_hrf**(-alpha) * np.exp(peak_hrf / beta)
			return [coef * c, alpha + 1., 1. / beta, 1.0]
		self._derivparams = [[glover2GammaDENS(peak_hrf[0], fwhm_hrf[0]), glover2GammaDENS(peak_hrf[1], fwhm_hrf[1], -dip)]]
		S = (self.eval(np.arange(0, 50.02, 0.02), 0) * 0.02).sum()
		for params in self._derivparams[0]: params[0] /= S
		for i in range(maxderiv):
			self._derivparams.append(self.derivParams(self._derivparams[-1]))
	derivParams = lambda s, paramlist, const=1 : reduce(list.__add__, [[[Coef*const*c*(alpha-1), alpha-1., nu, 1.0], [Coef*(-const)*c*nu, alpha, nu, 1.0]] for Coef, alpha, nu, c in paramlist])
	def eval(self, x, deriv = 0):
		_x = x * np.greater_equal(x, 0)
		value = zeros_like(x)
		for coef, alpha, nu, c in self._derivparams[deriv]:
			value += coef * ( c * np.power(_x, alpha-1.) * np.exp(-nu*_x) )
		return value

if __name__ == '__main__':
	from pylab import *
	from numpy import *
	h = MiniHRF()
	T = linspace(0,40, 200)
	figure()
	plot(h.eval(T)) # default : no deriv
	figure()
	plot(transpose([h.eval(T), h.eval(T, 1), h.eval(T, 2)]))
	figure()
	h = MiniHRF(peak_hrf=(1.4, 7.8), fwhm_hrf=(1.2, 3.35), dip=0.95)
	plot(h.eval(T))


# Cree une design matrix a partir d'une liste d'evenements.
# Construite salement en sub-samplant a 1/100 et en utilisant MiniHRF
# C'est + un exemple, car il faudrait que j'ecrive une vraie routine plus propre
def subsample_matrix_example(timegrid, factorlist, valueslist = None, deriv = False, duration = 0.5):
	# Todo : verifier la normalization de tous ces trucs
	# Todo verifier les unites (ici, tout est en secondes)
	from numpy import linspace, convolve, array
	hrf = MiniHRF()
	hrfcenti = hrf.eval(linspace(0, 30, 3000))
	hrfderivcenti = hrf.eval(linspace(0, 30, 3000), 1)
	timecenti = (timegrid * 100).astype(int)
	vectorlist = []
	for fa in factorlist:
		facenti = (fa * 100).astype(int)
		durcenti = int(duration * 100)
		vals = ones_like(fa) # a changer par les valeurs de valueslist
		M = zeros(timecenti.max()) # oversampled grid
		for s, e, v in zip(facenti, facenti+durcenti, vals):
			M[s:e] = v
		p = convolve(M, hrfcenti)
		Mp = p[timecenti]
		vectorlist.append(Mp)
		if deriv:
			p = convolve(M, hrfderivcenti)
			Mp = p[timecenti]
			vectorlist.append(Mp)
	return column_stack(vectorlist)

if __name__ == '__main__':
	timegrid = arange(140)
	from numpy.random import random
	factorlist = [random(size=6)*140 for x in range(12)]
	print factorlist
	X = subsample_matrix_example(timegrid, factorlist, None, deriv = True)
	#showX(X, namelist, timegrid, rpnames=mvt, deriv=True)
	figure()
	imshow(X, aspect='auto', interpolation='nearest')
	show()