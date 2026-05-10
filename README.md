<div align="center">
<img src="https://github.com/dagongren23/NIAH-Homogenization-Abaqus/blob/main/images/pluginbig.png" width="800" />
</div>

# NIAH-Homogenization-Abaqus
An Abaqus plug-in for asymptotic homogenization of periodic microstructures in Abaqus-Python

This tool seamlessly integrates the generation of Periodic Boundary Conditions (PBCs), macroscopic load case organization, and effective stiffness extraction into a single, user-friendly workflow within the commercial finite element environment.

## ✨ Features

- **Automated PBC Generation:** Systematically identifies and pairs boundary nodes, generating necessary linear equation constraints.
- **End-to-End Homogenization Analysis:** Implements the "three-step procedure" to evaluate effective mechanical properties.
- **Transverse Shear Stiffness Evaluation:** Employs non-homogeneous PBCs to capture the transverse shear stiffness (K) for Reissner–Mindlin plate models.
- **Broad Element Support:** Highly compatible with 3D solid, beam, and shell Representative Volume Elements (RVEs).
- **Stiffness Extraction & Visualization:** Automatically assembles equivalent constitutive matrices (Visualization module coming soon).

## 🛠️ Tested Environment

- **Abaqus:** 2020 (Highly likely compatible with 2017-2022 versions)
- **Python:** 2.7 (Abaqus built-in environment)

## 📥 Installation

1. Clone or download this repository.
2. Copy the entire plugin folder into your Abaqus plugins directory. Typically, this is located at:
   ```text
   C:\SIMULIA\CAE\plugins\2020\
   (Note: The exact path may vary depending on your Abaqus installation directory).
   ```
3. Restart Abaqus/CAE.

## 🚀 Usage

Access the tool via the top menu bar in Abaqus/CAE:
	Plug-ins → NIAH Homogenization
	(Screenshot of the plug-in interface)

1. Model Preparation
	Prepare a standard Abaqus input file (.inp) containing your meshed microstructure (e.g., beam_octet.inp).
		Geometric Requirements: The unit cell must be symmetric with respect to the Cartesian coordinate axes. Nodes on opposite boundary surfaces must be perfectly paired (1-to-1 correspondence), and the boundary faces must be parallel to the orthogonal coordinate system.
		Property Definitions: 
			For structures with uniform material properties, the .inp file only needs to contain the mesh discretization.
			For heterogeneous configurations involving multiple element types or complex section profiles, the input file must include the complete assembly and material definitions (which can be readily exported via the Abaqus GUI).

2. GUI Configuration
	Through the plug-in GUI, specify the following parameters:
		File Path: Directory of the prepared .inp file.
		Analysis Type: Choose between 3D homogenization or Shell homogenization.
		Primary Element Type: The main element type used in your mesh (e.g., C3D8, B31, S3).
		Reference Node: Coordinates of the rigid reference node.
		Non-periodic Directions: (Required only for plate/shell homogenization).
		
3. Execution Steps
	Once the GUI is configured, execute the workflow:
		Pre-processing: Click to run the pre-processing module. The script will automatically read the .inp file, classify boundary nodes, and apply constraints.
		Homogenization Solver: After pre-processing is complete, run the solver. The computation will proceed automatically in the background. Upon completion, 
			a .txt file containing the equivalent effective properties (Stiffness Tensor) will be generated in the "your input file directory\\NIAH_CH_txt".
		Stiffness Visualization: (Under Development / Coming Soon) A standalone module to generate 3D directional Young's modulus distribution surfaces and 2D radar charts for ABD matrices.
		
## 📄 CitationIf 

you find this plug-in helpful in your research, please consider citing our work:

@article{YourCitationKey2026,
  title={A computational framework for asymptotic homogenization of periodic microstructures: unified implementation for three-dimensional and Reissner–Mindlin plate homogenization},
  author={Liu, Zhihui and others},
  journal={Composite Structures},
  year={2026},
  publisher={Elsevier}
}(We will update the citation details once the paper is officially published)

## ✉️ Contact
For any questions, issues, or source code requests for secondary development, please feel free to reach out:
Email: 320514030@mail.dlut.edu.cn
