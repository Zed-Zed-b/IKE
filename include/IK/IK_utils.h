#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <sys/stat.h>
#include <fstream>

namespace IFlib {
    
    template <typename T, size_t Alignment>
    struct aligned_allocator
    {
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using propagate_on_container_move_assignment = std::true_type;
        using is_always_equal = std::true_type;

        template <typename U>
        struct rebind
        {
            using other = aligned_allocator<U, Alignment>;
        };

        aligned_allocator() noexcept = default;
        template <typename U>
        aligned_allocator(const aligned_allocator<U, Alignment> &) noexcept {}

        T *allocate(size_type n)
        {
            if (n > std::numeric_limits<size_type>::max() / sizeof(T))
            {
                throw std::bad_alloc();
            }
            void *ptr = aligned_alloc(Alignment, n * sizeof(T));
            if (!ptr)
                throw std::bad_alloc();
            return static_cast<T *>(ptr);
        }

        void deallocate(T *p, size_type n) noexcept
        {
            free(p);
        }
    };

    template <typename T, size_t Align1, typename U, size_t Align2>
    bool operator==(const aligned_allocator<T, Align1> &, const aligned_allocator<U, Align2> &) noexcept
    {
        return Align1 == Align2;
    }

    template <typename T, size_t Align1, typename U, size_t Align2>
    bool operator!=(const aligned_allocator<T, Align1> &a, const aligned_allocator<U, Align2> &b) noexcept
    {
        return !(a == b);
    }

    // 工具函数
    template <typename T = uint32_t>
    struct SparseDataImpl
    {
        // 数据存储容器
        std::vector<T, aligned_allocator<T, 64>> data_flat;
        std::vector<T *> data_ptrs;                       

        size_t n_features; // 特征维度
        size_t n_samples;  // 样本数量
    };

    // 当不指定类型时，默认使用 uint32_t
    using SparseData = SparseDataImpl<>;
    using SparseDataUINT8 = SparseDataImpl<uint8_t>;
    using SparseDataUINT16 = SparseDataImpl<uint16_t>;

    size_t inline calculateCodeSize(size_t t, size_t psi) {
        size_t code_size;
        
        if (psi == 2) {
            code_size = (t + 7) / 8;
        } else if (psi <= 4) {
            code_size = (t + 3) / 4;
        } else if (psi <= 16) {
            code_size = (t + 1) / 2;
        }else{
            code_size = t;
        }

        return code_size;
    }

    bool inline saveVectorToBinaryFile(const SparseDataUINT8 &data, const std::string &filename)
    {
        std::ofstream outFile(filename, std::ios::binary);
        if (!outFile)
        {
            std::cerr << "无法打开文件进行写入: " << filename << std::endl;
            return false;
        }

        outFile.write(reinterpret_cast<const char *>(data.data_flat.data()), data.data_flat.size());

        if (!outFile)
        {
            std::cerr << "写入文件时发生错误: " << filename << std::endl;
            outFile.close();
            return false;
        }

        outFile.close();
        return true;
    }

    bool inline loadVectorFromBinaryFile(SparseDataUINT8 &data, const std::string &filename)
    {
        std::ifstream inFile(filename, std::ios::binary);
        if (!inFile)
        {
            std::cerr << "无法打开文件进行读取: " << filename << std::endl;
            return false;
        }

        inFile.seekg(0, std::ios::end);
        std::streamsize fileSize = inFile.tellg();
        inFile.seekg(0, std::ios::beg);

        if (static_cast<size_t>(fileSize) != data.data_flat.size())
        {
            std::cerr << "文件大小与预分配向量大小不匹配: "
                      << fileSize << " != " << data.data_flat.size() << std::endl;
            inFile.close();
            return false;
        }

        inFile.read(reinterpret_cast<char *>(data.data_flat.data()), fileSize);

        if (!inFile)
        {
            std::cerr << "读取文件时发生错误: " << filename << std::endl;
            inFile.close();
            return false;
        }

        inFile.close();
        return true;
    }

    bool inline fileExists(const std::string &filename)
    {
        struct stat buffer;
        return (stat(filename.c_str(), &buffer) == 0);
    }
}