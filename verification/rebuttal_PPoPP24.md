
We appreciate the reviewers comments and constructive suggestions. We will incorporate our answers in our revision.

#### Common Questions:

#### Q1: Portability of CommBench (Rev A/C).

1) “The portability of the benchmark is in question, since the software stack is specific.”
2) “How is the portability to other GPUs achieved, as the system seems to rely only on NCCL? How much work is it to include a new communication fabric?”


CommBench is portable across Nvidia, AMD, and Intel GPUs (see Table 1). CommBench also has MPI, NCCL, RCCL, and IPC implementations (see Figs 7-9). The compiler and runtime flags are shown in Table 3. Supporting a new communication fabric requires implementing a few CommBench primitives (e.g., point-to-point communication and synchronization/fences) using the fabric’s operations. 

#### Q2: Why CommBench (Rev A/C)?

1) "If the user is looking to determine the performance of a specific communication library as used by their applications, might they not be better of specifically using those functions?"
2) "What results could have not been achieved with conventional benchmark suites?"

Conventional benchmarks measure the end-to-end performance of functions. On a hierarchical network, end-to-end communication is often made up of separate communications across different interconnects.  CommBench is designed for isolating  the performance of a specific interconnect. An example result that conventional end-to-end benchmarks cannot reveal is that current communication libraries do not take full advantage of multi-GPU, multi-NIC node architectures.

