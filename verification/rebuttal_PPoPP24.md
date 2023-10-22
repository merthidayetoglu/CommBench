
We appreciate the reviewers comments and constructive suggestions. We will incorporate our answers in our revision.

###Common Questions:

#### Q1: Portability of CommBench (Rev A/C).

CommBench is portable across Nvidia, AMD, and Intel GPUs (see Table 1). CommBench also has MPI, NCCL, RCCL, and IPC implementations (see Figs 7-9). The compiler and runtime flags are shown in Table 3. Supporting a new communication fabric requires implementing a few CommBench primitives (e.g., point-to-point communication and synchronization/fences) using the fabric’s operations. 

#### Q2: Why CommBench (Rev A/C)?

