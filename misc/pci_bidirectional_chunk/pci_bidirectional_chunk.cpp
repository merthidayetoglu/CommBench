
#include <array>
#include <iomanip>
#include <iostream>
#include <level_zero/ze_api.h>
#include <limits>
#include <sycl/sycl.hpp>

constexpr int n_commands_lists = 1 + 7;
// Index of the command list used by each chunk ( 0 == main copy, 1 first link engine...)
// None can reach more than 64 GB/s
template <typename T>
long foo(size_t N, std::string dir, std::vector<int> index_h2d, std::vector<int> index_d2h,
         int n_repetitions = 1)
{

    long ze_main_mintime = std::numeric_limits<long>::max();
    long ze_secondary_mintime = std::numeric_limits<long>::max();

    // Always create all the possible command queue, and allocate all memory
    // But we will use only what we need
    sycl::queue Q;
    auto *h0 = sycl::malloc_host<T>(N, Q);
    auto *d0 = sycl::malloc_device<T>(N, Q);

    auto *h1 = sycl::malloc_host<T>(N, Q);
    auto *d1 = sycl::malloc_device<T>(N, Q);

    ze_context_handle_t hContext =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(Q.get_context());
    ze_device_handle_t hDevice =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(Q.get_device());

    std::array<ze_command_list_handle_t, n_commands_lists> hCommandList;
    {
        ze_command_queue_desc_t cmdQueueDesc = {};
        cmdQueueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        cmdQueueDesc.ordinal = 1; // Main copy engines ordianl
        cmdQueueDesc.index = 0;
        zeCommandListCreateImmediate(hContext, hDevice, &cmdQueueDesc, &hCommandList[0]);

        cmdQueueDesc.ordinal = 2; // ordinal of the 7 aditional copy engine
        for (int i = 0; i < 7; i++)
        {
            cmdQueueDesc.index = i;
            zeCommandListCreateImmediate(hContext, hDevice, &cmdQueueDesc, &hCommandList[i + 1]);
        }
    }
    ze_event_pool_handle_t hEventPool;
    std::array<ze_event_handle_t, n_commands_lists> hEvent;
    {
        ze_event_pool_desc_t eventPoolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                              ZE_EVENT_POOL_FLAG_HOST_VISIBLE, n_commands_lists};

        zeEventPoolCreate(hContext, &eventPoolDesc, 0, nullptr, &hEventPool);
        ze_event_desc_t eventDesc = {};
        eventDesc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
        eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;

        for (int i = 0; i < n_commands_lists; i++)
        {
            eventDesc.index = i;
            zeEventCreate(hEventPool, &eventDesc, &hEvent[i]);
        }
    }

    for (int r = 0; r < n_repetitions; r++)
    {

        int event_used = 0;
        auto s0 = std::chrono::high_resolution_clock::now();
        if (dir == "H2D" || dir == "H2D+D2H")
        {
            int i = 0;
            size_t chunk_size = N / index_h2d.size();
            for (auto index : index_h2d)
            {
                zeCommandListAppendMemoryCopy(hCommandList[index], h0 + i * chunk_size, d0 + i * chunk_size,
                                              chunk_size * sizeof(T), hEvent[event_used], 0, nullptr);
                i++;
                event_used++;
            }
        }
        if (dir == "D2H" || dir == "H2D+D2H")
        {
            int i = 0;
            size_t chunk_size = N / index_d2h.size();
            for (auto index : index_d2h)
            {
                zeCommandListAppendMemoryCopy(hCommandList[index], d1 + i * chunk_size, h1 + i * chunk_size,
                                              chunk_size * sizeof(T), hEvent[event_used], 0, nullptr);
                i++;
                event_used++;
            }
        }

        for (int i = 0; i < event_used; i++)
            zeEventHostSynchronize(hEvent[i], std::numeric_limits<uint64_t>::max());

        const auto e0 = std::chrono::high_resolution_clock::now();
        const auto time = std::chrono::duration_cast<std::chrono::microseconds>(e0 - s0).count();
        ze_secondary_mintime = std::min(ze_secondary_mintime, time);

        for (int i = 0; i < event_used; i++)
            zeEventHostReset(hEvent[i]);
    }

    int payload = 1;
    if (dir == "H2D+D2H")
        payload = 2;

    std::cout << dir << " PCI BW: " << std::fixed << std::setprecision(2)
              << 1E-3 * N * payload * sizeof(T) / ze_secondary_mintime << " GBytes/s" << std::endl;

    for (int i = 0; i < n_commands_lists; i++)
    {
        zeEventDestroy(hEvent[i]);
        zeCommandListDestroy(hCommandList[i]);
    }

    sycl::free(h0, Q);
    sycl::free(d0, Q);
    sycl::free(h1, Q);
    sycl::free(d1, Q);
    return ze_secondary_mintime;
}

int main(int argc, char *argv[])
{
    std::vector<int> index_h2d;
    std::vector<int> index_d2h;

    {
        std::vector<int> *des;
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "--h2d")
                des = &index_h2d;
            else if (arg == "--d2h")
                des = &index_d2h;
            else
                des->push_back(stoi(arg));
        }
    }

    const int gcd = index_h2d.size() * index_h2d.size();
    const size_t bytes = (1e9 / gcd) * gcd;
    const size_t N = bytes / sizeof(double);
    const int n_repetitions = 100;

    auto h2d_time = foo<double>(N, "H2D", index_h2d, index_d2h, n_repetitions);
    auto d2h_time = foo<double>(N, "D2H", index_h2d, index_d2h, n_repetitions);
    auto both_time = foo<double>(N, "H2D+D2H", index_h2d, index_d2h, n_repetitions);

    // We assume 50 versus 65 bw difference
    if (both_time > (std::max(d2h_time, h2d_time)) * 0.33)
    {
        std::cout << "Failing. Not a full duplex..." << std::endl;
    std:
        exit(1);
    }
}
