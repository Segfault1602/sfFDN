#include "allocation_counter.h"

#include <cstdlib>
#include <new>

namespace
{
thread_local bool count_allocations = false;
thread_local size_t allocation_count = 0;

void RecordAllocation() noexcept
{
    if (count_allocations)
    {
        ++allocation_count;
    }
}

void* Allocate(size_t size)
{
    RecordAllocation();
    if (void* allocation = std::malloc(size == 0 ? 1 : size))
    {
        return allocation;
    }
    throw std::bad_alloc();
}

void* AllocateAligned(size_t size, size_t alignment)
{
    RecordAllocation();
    const size_t aligned_size = ((size == 0 ? 1 : size) + alignment - 1) / alignment * alignment;
    if (void* allocation = std::aligned_alloc(alignment, aligned_size))
    {
        return allocation;
    }
    throw std::bad_alloc();
}
} // namespace

void* operator new(size_t size)
{
    return Allocate(size);
}

void* operator new[](size_t size)
{
    return Allocate(size);
}

void operator delete(void* allocation) noexcept
{
    std::free(allocation);
}

void operator delete[](void* allocation) noexcept
{
    std::free(allocation);
}

void operator delete(void* allocation, size_t) noexcept
{
    std::free(allocation);
}

void operator delete[](void* allocation, size_t) noexcept
{
    std::free(allocation);
}

void* operator new(size_t size, std::align_val_t alignment)
{
    return AllocateAligned(size, static_cast<size_t>(alignment));
}

void* operator new[](size_t size, std::align_val_t alignment)
{
    return AllocateAligned(size, static_cast<size_t>(alignment));
}

void operator delete(void* allocation, std::align_val_t) noexcept
{
    std::free(allocation);
}

void operator delete[](void* allocation, std::align_val_t) noexcept
{
    std::free(allocation);
}

void operator delete(void* allocation, size_t, std::align_val_t) noexcept
{
    std::free(allocation);
}

void operator delete[](void* allocation, size_t, std::align_val_t) noexcept
{
    std::free(allocation);
}

namespace sfFDNTest
{

ScopedAllocationCounter::ScopedAllocationCounter() noexcept
{
    allocation_count = 0;
    count_allocations = true;
}

ScopedAllocationCounter::~ScopedAllocationCounter()
{
    count_allocations = false;
}

size_t ScopedAllocationCounter::Count() const noexcept
{
    return allocation_count;
}

} // namespace sfFDNTest
