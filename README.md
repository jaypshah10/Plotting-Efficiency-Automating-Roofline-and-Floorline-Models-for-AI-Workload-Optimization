
# Plotting-Efficiency-Automating-Roofline-and-Floorline-Models-for-AI-Workload-Optimization

This research aims to develop an automated framework for roofline and floorline plot analysis of deep neural network (DNN) accelerators to make analysis of a ML/AI workload on a given architecture faster and more efficient. The framework will evaluate trade-offs in energy efficiency, memory bandwidth, and computational power. By integrating roofline and floorline analysis, it provides insights for hardware-software co-design of ML workloads, considering computational intensity, memory access, and bit-precision. The tool will streamline trade-off analysis and support diverse hardware comparisons. Leveraging existing energy estimation and performance analysis methodologies, the project extends to floorline analysis for comprehensive DNN accelerator evaluation, contributing to more efficient and accessible AI systems.

## Steps to do Model Analysis

Edit Main File -> *Plot_analysis.ipynb*

### Step 1. Calculate Per layer CI (Computational Intensity) and OI (Operational Intensity) using  Workload input

**For Vision Transformers**

Example Parameters for ViT

	 image_size = 32  # Input image size (e.g., 224x224 for ViT-Base)
	 patch_size = 4  # Patch size (e.g., 16x16 patches)
	 num_input_channels = 3  # Input channels (e.g., RGB image)
     embedding_dim = 16  # Embedding dimension
     num_heads = 4  # Number of attention heads
     feedforward_dim = 32  # Feedforward network dimension
     output_dim = 10  # Output dimension (e.g., 1000 classes for classification)

Vision Transformer (ViT) CI and OI Calculator Output
	 
	FLOPs Breakdown: 
	Patch Embedding FLOPs: 99,328 
	MHSA Layer FLOPs: 394,240 
	FFN Layer FLOPs: 134,144 
	Classification Head FLOPs: 330 
	Operation Intensities: 
	Patch Embedding Layer OI: 6.0625 
	MHSA Layer OI: 4.5833 
	FFN Layer OI: 5.4583 
	Classification Head OI: 0.4853 
	CI per layer: [99328, 394240, 134144, 330] 
	OI per layer: [6.0625, 4.583333333333333, 5.458333333333333, 0.4852941176470588]
	
**For DNN Workloads**

Example usage: Define layer parameters directly

Args:

layer (int): Layer number
input_rows (int): Number of input rows

input_cols (int): Number of input columns

output_rows (int): Number of output rows

output_cols (int): Number of output columns

kernel_size (int): Kernel size

input_channels (int): Number of input channels

output_channels (int): Number of output channels

	layers_input = [
	(1,  224,  224,  55,  55,  121,  3,  96),
	(2,  27,  27,  27,  27,  25,  96,  256),
	(3,  13,  13,  13,  13,  9,  256,  384),
	(4,  13,  13,  13,  13,  9,  384,  384),
	(5,  13,  13,  13,  13,  9,  384,  256),
	(6,  9216,  1,  4096,  1,  1,  9216,  1),
	(7,  4096,  1,  4096,  1,  1,  4096,  1),
	(8,  4096,  1,  1000,  1,  1,  4096,  1),
	]

Deep Neural Network Layer Information Collector Output

	Operational Intensity (OI) for All Layers: 
	Layer 1: OI = 110.78 
	Layer 2: OI = 257.11 
	Layer 3: OI = 75.30 
	Layer 4: OI = 76.97 
	Layer 5: OI = 75.30 
	Layer 6: OI = 0.22 
	Layer 7: OI = 0.50 
	Layer 8: OI = 0.12 
	Computational Intensity (CI) for All Layers: 
	Layer 1: CI = 210,830,400 
	Layer 2: CI = 895,795,200 
	Layer 3: CI = 299,040,768 
	Layer 4: CI = 448,561,152 
	Layer 5: CI = 299,040,768 
	Layer 6: CI = 75,497,472 
	Layer 7: CI = 33,554,432 
	Layer 8: CI = 8,192,000 
	CI Array: [210830400, 895795200, 299040768, 448561152, 299040768, 75497472, 33554432, 8192000] 
	OI Array: [110.78, 257.11, 75.3, 76.97, 75.3, 0.22, 0.5, 0.12]

The CI (Computational Intensity) and OI (Operational Intensity) array returned have to be used for the plot.

### Step 2. Floorline Plot

 - Define workload inputs in the *workload_inputs* function as shown below

*workload_inputs(workload_name, CIs, annotation_OIs)*

	workload = workload_inputs(
	"AlexNet",
	CIs=np.array([210830400,  895795200,  299040768,  448561152,  299040768,  75497472,  33554432,  8192000]),
	annotation_OIs=np.array([109.8,  257.11,  75.29,  76.9,  75.2,  0.5,  0.5,  0.5])
	)

 - Define Architecture parameter inputs in the *set_architecture_params*   function as shown below

*set_architecture_params(arch_name, E_op_pJ, E_R)*
where, E_op is Energy per operation, E_R is ratio of E_m/E_op.
E_m is the Energy per memory access.

	set_architecture_params("Dig",  2000,  0.8)  # Digital Accelerator (E_op in pJ)
	set_architecture_params("PIM",  2,  0.0075)  # Processor-In-Memory (E_op in pJ)
	set_architecture_params("PNM",  200,  0.075)  # Processor-Near-Memory (E_op in pJ)

 - For Floorline Plotting pass workload and arch to *plot_analysis*
   function as shown below

*plot_analysis(workload, arch_names)*

	plot_analysis(workload,  ["Dig"])
	plot_analysis(workload,  ["PIM"])
	plot_analysis(workload,  ["PNM"])

### Step 2. Roofline Plot

 - Define workload inputs in the *workload_inputs* function as shown below

*workload_inputs(workload_name, workload_OIs)*

	workload = workload_inputs(
	"AlexNet",
	workload_OIs=[109.8,  257.11,  75.29,  76.9,  75.2,  0.5,  0.5,  0.5]
	)
	

 - Define Architecture parameter inputs in the *set_architecture_params*   function as shown below

*set_architecture_params(arch_name, peak_performance_tops, memory_bandwidth_tbps, bit_precision)*

	set_architecture_params("Dig", peak_performance_tops=275, memory_bandwidth_tbps=1.2, bit_precision=16)  # Digital Accelerator
	set_architecture_params("PIM", peak_performance_tops=400, memory_bandwidth_tbps=2.9, bit_precision=16)  # Processor-In-Memory
	set_architecture_params("PNM", peak_performance_tops=400, memory_bandwidth_tbps=2.2, bit_precision=16)  # Processor-Near-Memory

 - For Roofline Plotting pass workload and arch to *plot_analysis*
   function as shown below

*plot_analysis(workload, arch_names)*

	plot_analysis(workload,  ["Dig"])
	plot_analysis(workload,  ["PIM"])
	plot_analysis(workload,  ["PNM"])


