
We appreciate the reviewers comments and constructive suggestions. We will incorporate our answers in our revision.

#### Common Questions (Rev A/C):

#### Q1: Portability of CommBench.

1) “The portability of the benchmark is in question, since the software stack is specific.”
2) “How is the portability to other GPUs achieved, as the system seems to rely only on NCCL? How much work is it to include a new communication fabric?”

CommBench is portable across Nvidia, AMD, and Intel GPUs (see Table 1). CommBench also has MPI, NCCL, RCCL, and IPC implementations (see Figs 7-9). The compiler and runtime flags are shown in Table 3. Supporting a new communication fabric requires implementing a few CommBench primitives (e.g., point-to-point communication and synchronization/fences) using the fabric’s operations. 

#### Q2: Why CommBench?

1) "If the user is looking to determine the performance of a specific communication library as used by their applications, might they not be better of specifically using those functions?"
2) "What results could have not been achieved with conventional benchmark suites?"

Conventional benchmarks measure the end-to-end performance of functions. On a hierarchical network, end-to-end communication is often made up of separate communications across different interconnects.  CommBench is designed for isolating  the performance of a specific interconnect. An example result that conventional end-to-end benchmarks cannot reveal is that current communication libraries do not take full advantage of multi-GPU, multi-NIC node architectures.

#### Reviewer A:

#### Q1.1: Rationale for the p, g, k groupings is unclear.
The p value is the total number of GPUs the application uses. The g value is the number of GPUs that form a group with more efficient communication between them than across groups. The g value is determined by the machine topology. The k-value is the number of GPUs in each group to actually work more closely together to shift some of the global communication into intra-group communication. The rail/fan/dense patterns reflect the communication needs of the applications. In each pattern, one can vary the k-value from 1 to g to regulate the amount of global communication shifted into intra-group communication.

#### Q1.2: Particularly, for Frontier (Page 4: bottom right column), why use k=2?
In the Frontier example, k=2 because the 8 GPUs on a node are physically divided into four groups of 2, where the 2 GPU in a group are located in the same (two-die) device that is binded to a single NIC (see Figure 2 (e)). This means the NIC is shared by two GPUs and k=2 test the saturation with of a single NIC with two GPUs in isolation.

#### Q2: Expand on the terminology used in Section 5.1.3: Equation 1.
We will expand the terminology for the measurement payload (5.1.3).

#### Q3: Will your benchmarking tool be made publicly available?
CommBench is publicly available, open source, and documented. We will include the link in the final version.

#### Reviewer B:

#### Q1: Needs a section to summarize crucial findings or results.
We will include a section that lists our crucial findings:

1) The software overhead of MPI and NCCL varies significantly across the levels of the communication hierarchy.
2) MPI implementation does not saturate the GPU memory bandwidth in self (in-memory) communications.
3) Within nodes, MPI point-to-point communication does not overlap and unnecessarily prolongs the communication time. Such software-drawn boundaries should be eliminated.
4) Across Nodes, MPI has a lower latency with short message sizes and NCCL has a higher bandwidth in large message sizes. 
5) The GPU-NIC bindings can be static or dynamic in non-obvious ways. Understanding the association is crucial for optimizing collective communications.

#### Q2: How are the GPU-NIC bindings?
The GPU-NIC association can be static or dynamic (depends on how the underlying libraries handle it). MPI has static bindings (block or round-robin), and NCCL has dynamic binding (depending on the workload and the system). See Section 5.3.2 for more detail. If static, each node in the supercomputer has the same logical binding.

#### Q3: Presentation of results.
We will list these crucial findings at the beginning of the evaluations, and highlight these five points clearly when explaining the performance figures.

#### Q4: Optimization of Gather and Scatter.
It can be considered outside of the scope or authors should motivate it as well why they included in the study.
Striping of Gather and Scatter functions are included as an example action item towards hierarchical optimization of collective communications. We implement and validate the optimization using CommBench. We will integrate this short section into the paper more clearly.

#### Reviewer C:

#### Q1: Are there any overheads?
We minimize the measurement overhead of CommBench. Specifically, we exclude setup costs and global synchronization from the measurement. The remaining overhead is a few function calls on the order of ~35 ns which is insignificant compared to network latency (2 - 200 microseconds). 

#### Q2: Ease of use compared to other benchmark suites.
CommBench is programmed into a single header file that works across compilers and can be easily included in the application code. Moreover, the group-to-group patterns are pre-implemented with the proposed API and parameterized with command-line parameters (similar to other benchmarks).

#### Q3: Accuracy
Other benchmarks measure end-to-end collective function time. Our benchmark measures group-to-group time as well.  Our end-to-end measurements are as accurate as other approaches; beyond that there is no way to make a meaningful comparison.

#### Q4: Workflow of evaluating a new system.
We will describe the steps that one needs to take to repeat the measurements reported in this paper. In short, 1) compilation with specific GPU port, 2) setting up the parameters depending on the system and experiments, and 3) post-processing CommBench’s output with a script and plotting the results. Typically, it takes about 30 minutes to reproduce our results per system. We will write the instruction with more detail in the AD/AE appendix.

#### Q5: Minor comment on terminology.
The reviewer is correct. We will change ”MPI rank” to “MPI process”.

#### Reviewer D:

#### Q1: Grouping is conceptual.
The reviewer raised an insightful point. We defined the proposed group-to-group pattern parametrization for convention. Those patterns do extract the physical and logical mismatch at specific levels in the network hierarchy (e.g., GPU-to-NIC).

#### Q2: API.
The parameterized patterns are currently implemented as benchmarks rather than part of the API. The API can be used to configure similar parameterized benchmarks.










