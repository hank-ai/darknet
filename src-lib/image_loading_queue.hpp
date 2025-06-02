#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <atomic>
#include <memory>
#include <optional>
#include "darknet_internal.hpp"

namespace Darknet
{
    /// Task for loading a batch of images
    struct ImageLoadTask
    {
        load_args args;
        int thread_id;
        int batch_start;
        int batch_size;
    };

    /// Result from loading a batch of images
    struct ImageLoadResult
    {
        data loaded_data;
        int thread_id;
        bool success;
    };

    /// Thread-safe queue for image loading tasks
    template<typename T>
    class ThreadSafeQueue
    {
    private:
        mutable std::mutex mutex_;
        std::condition_variable cond_var_;
        std::queue<T> queue_;
        size_t max_size_;
        std::atomic<bool> shutdown_{false};

    public:
        explicit ThreadSafeQueue(size_t max_size = std::numeric_limits<size_t>::max())
            : max_size_(max_size) {}

        ~ThreadSafeQueue()
        {
            shutdown();
        }

        /// Add item to queue, blocks if queue is full
        bool enqueue(T item)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            
            // Wait until there's space or shutdown
            cond_var_.wait(lock, [this] {
                return queue_.size() < max_size_ || shutdown_.load();
            });

            if (shutdown_.load())
            {
                return false;
            }

            queue_.push(std::move(item));
            cond_var_.notify_one();
            return true;
        }

        /// Remove item from queue, blocks if queue is empty
        std::optional<T> dequeue()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            
            // Wait until there's an item or shutdown
            cond_var_.wait(lock, [this] {
                return !queue_.empty() || shutdown_.load();
            });

            if (shutdown_.load() && queue_.empty())
            {
                return std::nullopt;
            }

            T item = std::move(queue_.front());
            queue_.pop();
            cond_var_.notify_one(); // Notify producers that there's space
            return item;
        }

        /// Try to remove item without blocking
        std::optional<T> try_dequeue()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            
            if (queue_.empty())
            {
                return std::nullopt;
            }

            T item = std::move(queue_.front());
            queue_.pop();
            cond_var_.notify_one();
            return item;
        }

        size_t size() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }

        bool empty() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }

        void shutdown()
        {
            shutdown_.store(true);
            cond_var_.notify_all();
        }

        bool is_shutdown() const
        {
            return shutdown_.load();
        }

        void clear()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::queue<T> empty;
            std::swap(queue_, empty);
            cond_var_.notify_all();
        }
    };

    /// Image loading thread pool manager
    class ImageLoadingThreadPool
    {
    private:
        std::vector<std::thread> workers_;
        ThreadSafeQueue<ImageLoadTask> task_queue_;
        ThreadSafeQueue<ImageLoadResult> result_queue_;
        std::atomic<bool> stop_{false};
        const int num_threads_;
        std::atomic<int> pending_results_{0};

        /// Worker thread function
        void worker_thread(int thread_id);

    public:
        explicit ImageLoadingThreadPool(int num_threads = 6, size_t max_queue_size = 10);
        ~ImageLoadingThreadPool();

        /// Submit a batch loading task
        bool submit_batch(const load_args& args);

        /// Get loaded batch (blocks until ready)
        data get_loaded_batch();

        /// Check if results are available
        bool has_results() const { return !result_queue_.empty(); }

        /// Shutdown the thread pool
        void shutdown();

        /// Get number of pending tasks
        size_t pending_tasks() const { return task_queue_.size(); }

        /// Get number of ready results
        size_t ready_results() const { return result_queue_.size(); }
    };

    /// Global image loading thread pool (singleton)
    ImageLoadingThreadPool& get_image_loading_thread_pool();

    /// Initialize the global thread pool
    void initialize_image_loading_thread_pool(int num_threads);

    /// Shutdown the global thread pool
    void shutdown_image_loading_thread_pool();
}