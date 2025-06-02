#include "image_loading_queue.hpp"
#include "data.hpp"
#include "darknet_utils.hpp"
#include "darknet_cfg_and_state.hpp"
#include <chrono>
#include <iostream>

// Forward declaration for concat_datas function
namespace
{
    static inline data concat_datas(data *d, int n)
    {
        data out = {0};
        for (int i = 0; i < n; ++i)
        {
            data newdata = concat_data(d[i], out);
            Darknet::free_data(out);
            out = newdata;
        }
        return out;
    }
}

namespace Darknet
{
    namespace
    {
        static std::unique_ptr<ImageLoadingThreadPool> g_thread_pool;
        static std::mutex g_thread_pool_mutex;
    }

    void ImageLoadingThreadPool::worker_thread(int thread_id)
    {
        const std::string name = "image loading worker #" + std::to_string(thread_id);
        CfgAndState::get().set_thread_name(name);

        while (!stop_.load())
        {
            auto task_opt = task_queue_.dequeue();
            if (!task_opt.has_value())
            {
                break; // Queue was shut down
            }

            const auto& task = task_opt.value();
            
            // Create a local copy of args and modify for this thread's portion
            load_args local_args = task.args;
            local_args.n = task.batch_size;
            
            // IMPORTANT: Each thread needs its own data buffer
            local_args.d = task.args.d;
            
            // Allocate result data
            ImageLoadResult result;
            result.thread_id = thread_id;
            result.success = false;

            try
            {
                // Load the images for this thread's portion
                load_single_image_data(local_args);
                
                // Extract the loaded data
                if (local_args.d != nullptr)
                {
                    result.loaded_data = *local_args.d;
                    result.success = true;
                    
                    // Free the original buffer since we've copied the data
                    free(task.args.d);
                }
            }
            catch (const std::exception& e)
            {
                std::cerr << "Error in image loading thread " << thread_id 
                          << ": " << e.what() << std::endl;
                if (task.args.d != nullptr)
                {
                    free(task.args.d);
                }
            }

            // Submit result
            result_queue_.enqueue(std::move(result));
        }

        CfgAndState::get().del_thread_name();
    }

    ImageLoadingThreadPool::ImageLoadingThreadPool(int num_threads, size_t max_queue_size)
        : task_queue_(max_queue_size)
        , result_queue_(max_queue_size * 2) // Allow more results to queue up
        , num_threads_(num_threads)
    {
        workers_.reserve(num_threads);
        
        std::cout << "Creating " << num_threads 
                  << " permanent image loading threads with queue size " 
                  << max_queue_size << std::endl;

        for (int i = 0; i < num_threads; ++i)
        {
            workers_.emplace_back(&ImageLoadingThreadPool::worker_thread, this, i);
        }
    }

    ImageLoadingThreadPool::~ImageLoadingThreadPool()
    {
        shutdown();
    }

    bool ImageLoadingThreadPool::submit_batch(const load_args& args)
    {
        const int total_images = args.n;
        const int images_per_thread = (total_images + num_threads_ - 1) / num_threads_;
        int tasks_submitted = 0;
        
        // Submit tasks for each thread
        for (int i = 0; i < num_threads_; ++i)
        {
            ImageLoadTask task;
            task.args = args;
            task.thread_id = i;
            task.batch_start = i * images_per_thread;
            task.batch_size = std::min(images_per_thread, total_images - task.batch_start);
            
            if (task.batch_size <= 0)
            {
                break; // No more images to assign
            }

            // Allocate buffer for this thread's results
            task.args.d = (data*)xcalloc(1, sizeof(data));
            
            if (!task_queue_.enqueue(std::move(task)))
            {
                return false; // Queue was shut down
            }
            
            tasks_submitted++;
        }
        
        // Track how many results we're expecting
        pending_results_.store(tasks_submitted);
        
        return true;
    }

    data ImageLoadingThreadPool::get_loaded_batch()
    {
        // Get the number of results we're expecting
        int expected_results = pending_results_.load();
        
        // Collect results into an array like the original code
        data* thread_results = (data*)xcalloc(expected_results, sizeof(data));
        int results_received = 0;

        // Collect results from all threads
        while (results_received < expected_results)
        {
            auto result_opt = result_queue_.dequeue();
            if (!result_opt.has_value())
            {
                break; // Queue was shut down
            }

            const auto& result = result_opt.value();
            if (result.success && result.thread_id < expected_results)
            {
                thread_results[result.thread_id] = result.loaded_data;
                results_received++;
            }
        }

        // Use the same concat_datas function as the original code
        data combined = concat_datas(thread_results, expected_results);
        combined.shallow = 0;

        // Clean up the temporary thread results
        for (int i = 0; i < expected_results; ++i)
        {
            thread_results[i].shallow = 1;
            free_data(thread_results[i]);
        }
        free(thread_results);

        return combined;
    }

    void ImageLoadingThreadPool::shutdown()
    {
        stop_.store(true);
        task_queue_.shutdown();
        result_queue_.shutdown();

        for (auto& worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        workers_.clear();
    }

    ImageLoadingThreadPool& get_image_loading_thread_pool()
    {
        std::lock_guard<std::mutex> lock(g_thread_pool_mutex);
        if (!g_thread_pool)
        {
            throw std::runtime_error("Image loading thread pool not initialized");
        }
        return *g_thread_pool;
    }

    void initialize_image_loading_thread_pool(int num_threads)
    {
        std::lock_guard<std::mutex> lock(g_thread_pool_mutex);
        if (g_thread_pool)
        {
            std::cerr << "Warning: Image loading thread pool already initialized" << std::endl;
            return;
        }
        g_thread_pool = std::make_unique<ImageLoadingThreadPool>(num_threads);
    }

    void shutdown_image_loading_thread_pool()
    {
        std::lock_guard<std::mutex> lock(g_thread_pool_mutex);
        if (g_thread_pool)
        {
            g_thread_pool->shutdown();
            g_thread_pool.reset();
        }
    }
}