import numpy as np

diffusion_dict = {
        'BNS-800':{'eigvals':[0.086, 0.0026], 
                 'eigvecs':np.matrix([[0.06, 0.77], [-0.1, 0.23]])},
        'BNS-900':{'eigvals':[0.68, 0.036], 
                 'eigvecs':np.matrix([[0.02, 0.77], [-0.98, 0.23]])},
        'BNS-1000':{'eigvals':[5.4, 0.15], 
                 'eigvecs':np.matrix([[-0.1, 0.75], [1, 0.25]])},
        'BNS-1100':{'eigvals':[10.1, 0.5], 
                 'eigvecs':np.matrix([[-0.1, 0.75], [1, 0.25]])},
        'NCS':{'eigvals':[105, 4.4], 
                 'eigvecs':np.matrix([[1, 0.27], [-0.85, 0.73]])},
        }

initials_dict = {'B':'B2O3', 'N':'Na2O', 'C':'CaO', 'S':'SiO2'}
