# MPhys-Code
Code and plots relevant to work done during my MPhys Project.

The github is split into 2 parts; separating work done in Year 3 of my Physics Degree (First year for the project) and Year 4 of my degree (Second year for the project).

The following text provides context for each of the folders found in the github.

## Y3 Work
"Fibres" folder contains code relevant to working with Multimode Optical Fibres

"Modes Folder" folder contains png plots of all 42 fibre propagation invariant modes generated from the MATLAB code provided to us (found in the "Fibres" folder).

"NN Optimiser Tests" folder contains simple neural network optimisers for different equations. A useful starting point to build more complex models.

"T2 Work" folder contains work done during the second term of Y3, most of the work is with TorchOptics and getting used to coding with fields and neural networks.

"Phase Mask Work" contains work done primarily during T3, Y3. After being temporarily moved to working on free space propagation and phase masks for MPLCs instead of fibres. Tilt_Optimiser.ipynb contains the most recent code with a working axial optimiser for a tilt phase mask.

## Y4 Work
"Compressive Sampling" folder contains the most recent work on the project, as of updating the github this work is still ongoing. It contains notebooks relevant to the new technique we are using to model a multimode fibre: Compressive Sampling to generate a transmission matrix.

"Functions" folder is largely the most important in the entire github. It contains the functions / data reading code / models used by almost every notebook created in Y4. The python files in this folder are imported into all recent notebooks to save time and improve code readability.

"Images" folder contains a few custom 31x31 field images made in MS Paint. These were used to test the loss of data and how an image is transformed when travelling through a multimode optical fibre.

"Old Code" folder contains code relevant to optical fibres written either in Y3 or very early on in Y4.

"Original Model Work" contains most the work completed in Y4, it is focused on testing training models using our original method of generating the TM, before we concluded that it wasn't fit for real world applications.
