
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

