We thank the reviewers for their comments. We first address two common points:

**Q1: What is the connection of CommBench to applications? (Rev2, Rev4, Rev5)**

**A1:** CommBench provides measurement utilities for library implementers and application developers to make quick and reliable experiments towards tailored optimizations. Furthermore, CommBench API makes it easy to build collective operations that take advantage of a specific machine’s network.

For example, we observe rail--Figure 3(d)--pattern obtains a higher bandwidth across nodes than the fan--Figure 3(e)--pattern. The former utilizes multiple NICs per node and the latter chokes on a single NIC. Not yet included in the paper, we used this insight to implement Scatter operation with a few lines of code that runs approx. four times faster than MPI_Scatter function on Perlmutter. We tuned the performance by choosing IPC within nodes and NCCL across nodes using CommBench’s API.

We propose adding a short section to highlight this optimization insight and a key result. We already applied a striping algorithm to address **Rev3** and validated each optimization step with CommBench's benchmarking utilities.

**Q2: Why does CommBench not directly run collective functions? (Rev2&5)**

A2: Traditional benchmarks (OSU and NCCL tests) already run the collective functions that are partially listed in Table 2. We propose non-standard communication patterns that isolate the data movement across groups as discussed in Section 3. The proposed patterns are modified versions of the standard benchmarks (see Figure 3) to stress a specific level of the communication hierarchy by setting the group size.

Now we address individual comments:

**Rev 1:**
- We will fix the figure references and reported typos.
- In Section 5.3, CommBench runs on ThetaGPU—Figure 8(e)—correctly with MPI, but with approx. 10% of the theoretical cross-node bandwidth. After discussion with ALCF admins, we concluded that the MPI settings are not tuned properly to utilize NICs. Therefore we reported NCCL results only upon their recommendation. The problem has not been resolved. We propose including the missing MPI results (as they are) in the final version.

**Rev2:**
- Please refer to **Q1** and **Q2** for relation of CommBench to HPC applications. We will cite the survey paper from Laguna et al.
- This work focuses on data movement across groups only, i.e., with no computation. Therefore we do not currently support reduction operations. The implementer can easily add computation to extend the benchmark for reduction operation.
- We will use white borders in collective communication markers.

**Rev3:**
- Please refer **Q1**. Indeed, this optimization is efficient only for large messages because of the latency overhead.
- It is possible to perform additional experiments with CommBench to reverse engineer the internals of NCCL. Ideally, it is best to confirm with library developers. We are in contact with NCCL developers and will update the speculation upon their recommendation.
- “...groups intermediate in size…" refers to all processors in NUMA node, full node, rack, etc. These groups are of different sizes at different levels of the communication hierarchy. We will fix this sentence.
- CommBench will work as long as MPI or NCCL works. We would be happy to try out the pre-implemented patterns, or use CommBench’s API to implement new patterns for TPU nodes.
- We left the Omnidirectional Fan pattern undefined for brevity, i.e., we did not consider a practical use of it. Please see the text in Section 3 below Figure 5.

**Rev4:**
- We propose adding a case study to demonstrate CommBench for optimizing a library collective function that can be beneficial for many applications (please see **Q1**).

**Rev5:**
- Please refer to **Q1** for CommBench applicability and **Q2** for its relation to collective operations.
- CommBench's main utilities are 1) API to express a desired collective pattern and 2) build-in functions for taking accurate measurements with given pattern. We will make this clear.
- The logical topology depends on the library implementation, is not found in user guide, and communication libraries do not document their internal heuristics. Through discussions with system architects, we have learned that it is considered too much detail for a typical application developer.
- For the reason why not run collectives directly, please refer to **Q2**.
- We will cite the prior work from Chan, et al.
- We will remark that users can also use other system tools to determine the physical topology
- We will note that NCCL is open-source as of v2.3.
- CommBench exposed a serialization problem in MPICH-based MPI implementations. MPICH developers verified that their code does not overlap multiple IPC communications; it rather serializes them on a single GPU stream. We will include this finding in Section 5.4.
