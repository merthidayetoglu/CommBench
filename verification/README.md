We thank the reviewers for their comments. To address some common points:

**Q1: What is the connection of CommBench to applications? (Rev2, Rev4, Rev5)**

**A1:** CommBench provides utilities for guiding library implementers and application developers  to make quick and reliable experiments towards tailored algorithmic optimizations. Furthermore, the CommBench API supports building collective operations that take advantage of a specific machine’s network.

For example, we observe rail--Figure3(d)--pattern obtains a higher bandwidth than the fan--Figure 3 (e)--pattern across nodes. Because the former utilizes multiple NICs per node however, the latter chokes on a single NIC. We used this insight to implement Scatter operation that runs four times faster than MPI_Scatter function with a few lines of code using CommBench’s API. We tuned the performance by choosing IPC within nodes and NCCL across nodes using CommBench's API. 
 
We will add a section to describe these optimization insights and results. We already applied a striping algorithm to address Rev2 and validated each optimization step with CommBench's benchmarking utilities.
 
**Q2: Why does CommBench not directly run collective operations? (Rev2&5)**

A2: Traditional benchmarks (OSU and NCCL tests) already run the collective functions that are partially listed in Table 2. We use non-standard communication patterns that isolate the data movement across groups as discussed in Section 3.

The proposed patterns are reduced versions of the standard patterns (see Figure 3) to target a specific level of the communication hierarchy. We measure higher bandwidth lower latency with isolated group patterns than MPI_Scatter and MPI_Alltoall functions (Figure 8 and Figure 9).

Now we address individual comments.

Rev 1:
- We will fix the figure references and reported typos.
- In Section 5.3, CommBench runs on ThetaGPU—Figure 8(e)—without problem but the MPI bandwidth is low. After discussing this with ALCF admins, we speculate that there is a rank placement issue. The problem has not been resolved. But to address the comment, we will include the missing results in the final version.

Rev2:
- Please refer to Q1 and Q2 for relation of CommBench to HPC applications. We will cite the survey paper from Laguna et al.
- This work focuses on data movement across groups only, i.e., with no computation. Therefore we do not currently support reduction operations. The implementer can easily add computation to extend the benchmark to include the reduction operation.
- We will use white borders for collective communication marks.

Rev3:
- We will add a section to demonstrate the striping algorithm (please refer to Q1) that takes advantage of multiple NICs. We confirm that this optimization is only efficient for large messages because of the latency overhead.
- It is possible to perform additional experiments with CommBench to reverse engineer the internals of NCCL. Ideally, it is best to confirm with NCCL developers. We are in contact with NCCL engineers and will update speculation upon their recommendation.
- “... groups intermediate in size…" refers to nodes where each GPU is controlled by a single MPI process that runs on CPU. By processor, we mean GPU. We will fix this sentence.
- CommBench will work as long as MPI or NCCL works. We would be happy to try out the pre-implemented patterns, or use CommBench’s API to implement new patterns for TPU nodes.
- We left the Omnidirectional Fan pattern undefined for brevity, i.e., we did not consider a practical use of it. Please see the text in Section 3 below Figure 5.

Rev4:
- To address the motivation, we propose adding a section that demonstrates how CommBench is useful for optimizing a library collective function (please refer to Q1).
Rev5:
Please refer to Q1 for CommBench applicability and Q2 for its relation to collective operations.
- Those utilities are non-blocking measurement functions and a flexible API to construct a desired collective pattern. The goal of CommBench is to help developers to gain insights about the optimization opportunities through measurements.
- The logical topology depends on the library implementation, and is not found in user guide and communication libraries do not document their internal heuristics. Through discussions with system architects, we have learned that it is considered too much detail for a typical application developer.
For the reason why not run collectives directly, please refer to Q2.
- We will cite the prior work from Chan, et al.
- remark that users can also use other system tools to determine the physical topology
- note that NCCL is open-source in the final version.
- will note NCCL is not closed source nor proprietary.
- One interesting finding that CommBench exposed is the serialization problem with MPICH-based MPI implementations. We reported this finding to MPICH developers and verified that their code does not overlap multiple P2P communications because it serializes them on a single GPU stream. This is another good example of the applicability of CommBench.  We will describe this finding on Perlmutter in the final version as an example in Section 5.4.
