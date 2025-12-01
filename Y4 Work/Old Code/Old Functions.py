import numpy as np
import torch





def pim_matrix():
    """Creates the real space to pim space conversion matrix. Generated using external data for each PIM's shape."""
    pim = torch.column_stack((PIMs_torch[0].flatten(),PIMs_torch[1].flatten()))
    no_pims = PIMs_torch.size()[0]
    for i in np.arange(2, no_pims, 1):
        pim = torch.column_stack((pim, PIMs_torch[i].flatten()))
    return pim

def real_matrix():
    """Creates the pim space to real space conversion matrix. Generated using external data for each PIM's shape, from the pim_matrix() function."""
    return torch.transpose(torch.conj(pim_matrix()), dim0=0, dim1=1)

def beta_matrix(length=0):
    """Creates a diagonal array of speeds for each PIM that can travel through the MMF."""
    return torch.diag(torch.exp(1j * length * beta_torch.flatten()))

def TM(length=0):
    """Generates the full transmission matrix for a field travelling through a perfectly straight, ideal MMF."""
    real_to_pim = pim_matrix()
    pim_to_real = real_matrix()
    beta = beta_mat(length)
    return real_to_pim @ beta @ pim_to_real

# Make a propagate function to help save time (and less complexity) in the future
def propagate(input_field, length):
    """Takes an input field and the transmission matrix of a multimode optical fibre and finds the output after propagating for a given distance."""
    output_field_vector = TM(length) @ vectorise(input_field)
    output_field = unvectorise(output_field_vector,31)
    return output_field