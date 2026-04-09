#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace jakal::executors {

class HostThreadPool {
public:
    static HostThreadPool& instance() {
        static HostThreadPool pool;
        return pool;
    }

    [[nodiscard]] std::size_t worker_count() const {
        return workers_.size();
    }

    template <typename Func>
    void parallel_for(const std::size_t items, const std::size_t min_chunk, Func&& func) {
        if (items == 0u) {
            return;
        }

        const std::size_t concurrency = std::max<std::size_t>(1u, workers_.size() + 1u);
        if (concurrency <= 1u || items <= min_chunk) {
            func(0u, items);
            return;
        }

        const std::size_t chunk =
            std::max(min_chunk, (items + concurrency - 1u) / concurrency);
        const std::size_t task_count = (items + chunk - 1u) / chunk;
        if (task_count <= 1u) {
            func(0u, items);
            return;
        }

        std::mutex wait_mutex;
        std::condition_variable wait_cv;
        std::atomic<std::size_t> pending{task_count};

        auto complete = [&]() {
            if (pending.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
                std::lock_guard lock(wait_mutex);
                wait_cv.notify_one();
            }
        };

        std::size_t begin = 0u;
        for (std::size_t task_index = 0u; task_index + 1u < task_count; ++task_index) {
            const std::size_t task_begin = begin;
            const std::size_t task_end = std::min(items, task_begin + chunk);
            enqueue([task_begin, task_end, &func, &complete]() {
                func(task_begin, task_end);
                complete();
            });
            begin = task_end;
        }

        func(begin, items);
        complete();

        std::unique_lock lock(wait_mutex);
        wait_cv.wait(lock, [&]() {
            return pending.load(std::memory_order_acquire) == 0u;
        });
    }

private:
    HostThreadPool() {
        const auto threads = std::max(1u, std::thread::hardware_concurrency());
        workers_.reserve(threads > 1u ? threads - 1u : 0u);
        for (std::size_t index = 1u; index < threads; ++index) {
            workers_.emplace_back([this]() {
                worker_loop();
            });
        }
    }

    ~HostThreadPool() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    HostThreadPool(const HostThreadPool&) = delete;
    HostThreadPool& operator=(const HostThreadPool&) = delete;

    void enqueue(std::function<void()> task) {
        {
            std::lock_guard lock(mutex_);
            tasks_.push_back(std::move(task));
        }
        cv_.notify_one();
    }

    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [&]() {
                    return stopping_ || !tasks_.empty();
                });
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

}  // namespace jakal::executors
