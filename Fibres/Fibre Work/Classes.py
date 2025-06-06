# Imports
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchoptics
from torchoptics import Field, System, Param

import random

# Global Functions

# Make a vectorising function
def vectorise(field):
    x = field.shape[0]
    y = field.shape[1]
    return field.reshape(x*y,1)

# Make an unvectorising function
def unvectorise(field, dim):
    return field.reshape(dim,dim)

def Complex2HSV(z, rmin, rmax, hue_start=90):
    # get amplidude of z and limit to [rmin, rmax]
    amp = np.abs(z)
    amp = np.where(amp < rmin, rmin, amp)
    amp = np.where(amp > rmax, rmax, amp)
    ph = np.angle(z, deg=1)# + hue_start
    # HSV are values in range [0,1]
    h = (ph % 360) / 360
    s = 0.85 * np.ones_like(h)
    v = (amp -rmin) / (rmax - rmin)
    return hsv_to_rgb(np.dstack((h,s,v)))

def random_field(x,y):
    array = []
    for n in range(x):
        row = []
        for m in range(y):
            row.append(random.randint(0,1))
        array.append(row)
    return Field(torch.tensor(array, dtype=torch.cfloat))

def random_speckle_field_test(fibre):
    """Make a random speckle field containing a random set of PIMs."""
    field_as_PIMs = torch.rand((fibre.no_pims,1), dtype=torch.cfloat)
    field_as_real = torch.mm(fibre.pim_to_real, field_as_PIMs)
    field_norm = Field(field_as_real).normalise()
    print(fibre.grid_size)
    field_norm = unvectorise(field_norm, fibre.grid_size)
    return field_norm

# Make a fibre class that contains all the necessary info and functions that we need use use.
class Fibre:
    # Class global variables
    wl = 633e-9 # [m]
    grid_size = 31 # [pixels long (grid is square)]
    core_radius = 12.5e-6 # [m]
    core_radius_pix = 15 # [in pixels]
    dx = core_radius/core_radius_pix # Size of a pixel [m]
    phy_size = grid_size*dx # Physical size of the grid [m]

    # Initialise fibre
    def __init__(self, PIMs, beta):
        self.PIMs = PIMs
        self.beta = beta
        self.no_pims = PIMs.size(dim=0)

        pim = PIMs[0].reshape((PIMs.size(dim=1)**2),1)
        for i in np.arange(1,self.no_pims,1):
            pim = np.concatenate([pim, self.PIMs[i].reshape((PIMs.size(dim=1)**2),1)], axis=1)
        self.real_to_pim = torch.tensor(pim)
        self.pim_to_real =  torch.tensor(np.transpose(np.conj(pim)))

    # Functions relevant to a Fibre
    def TM(self, length):
        """Forms a transmission matrix, TM, based on externally generated PIMs and their corresponding phase velocities, built into the Fibre class.
        * length = Length of the MMF"""

        # Create the real space to PIM matrix
        pim = torch.column_stack((self.PIMs[0].flatten(),self.PIMs[1].flatten()))
        for i in np.arange(2,self.no_pims,1):
            pim = torch.column_stack((pim, self.PIMs[i].flatten()))

        # Take the conjugate transpose of the pim matrix to be used in returning the field back to pixel coords
        pim_dag = torch.transpose(torch.conj(pim), dim0=0, dim1=1)
    
        # Make a diagonal PIM to PIM TM (each term is exp(i*L*beta), where L is the length of the MMF)
        beta_arr = torch.diag(torch.exp(1j*length*self.beta.flatten()))
    
        # Combine each matrix to make the full TM for the MMF
        return pim @ beta_arr @ pim_dag


    # Make a propagate function to help save time (and less complexity) in the future
    def propagate(self, input_field, length):
        # Give option to use TorchOptics fields or not!
        output_field_vector = self.TM(length) @ vectorise(input_field).detach()
        return Field(unvectorise(output_field_vector,self.grid_size))
    
    #def forward(self, input_field, length):
        #return self.propagate(input_field, length)


class Field(torch.Tensor):
    """Class designed around a 2D electric field represented by a torch tensor."""
    def __init__(self, tensor):
        self = tensor

    def normalise(self):
        return self / torch.linalg.matrix_norm(self)

    def visualise(self, title=""):
        """Displays a visual plot of the field, using hsv colour mapping to demonstrate the fields phase (Hue) and amplitude (Value)."""
        # Set up plots
        fig, axs = plt.subplots(1,2, figsize=(10,5))

        # Plot the given field
        axs[0].imshow(Complex2HSV(self, 0, 0.065))

        # Colour bar
        V, H = np.mgrid[0:1:100j, 0:1:300j]
        S = np.ones_like(V)
        HSV = np.dstack((H,S,V))
        RGB = hsv_to_rgb(HSV)

        axs[1].imshow(RGB, origin="lower", extent=[0, 2*np.pi, 0, 1], aspect=15)

        axs[1].set_xticks([0, np.pi, 2*np.pi], ['0', '$\pi$', '$2\pi$'])
        axs[1].set_yticks([0, 1], ['0', '1'])

        axs[1].set_ylabel("Normalised Amplitude")
        axs[1].set_xlabel("Phase (rad.)")

        fig.show()