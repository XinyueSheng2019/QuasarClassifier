import celerite
# import numpy as np
import autograd.numpy as np
import pandas as pd
from celerite import terms

class DRW_kernel(terms.Term):
	parameter_names = ("log_a", "log_c")
	def __repr__(self):
		return "DRW_kernel({0.log_a}, {0.log_c})".format(self)
	def get_real_coefficients(self, params):
		log_a, log_c = params
		return np.exp(2.0*log_a), 1.0/np.exp(log_c)


kernel = DRW_kernel(log_a = 0.1, log_c=200)

print(kernel.get_parameter_dict())

