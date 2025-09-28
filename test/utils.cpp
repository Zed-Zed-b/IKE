#include "utils.h"

vector<vector<int32_t>> read_ivecs(const string &filename)
{
    ifstream in(filename, ios::binary);
    if (!in)
        throw runtime_error("无法打开文件: " + filename);

    vector<vector<int32_t>> data;

    while (true)
    {
        // 读取维度 (小端int32)
        int32_t dim;
        in.read(reinterpret_cast<char *>(&dim), 4);
        if (in.gcount() != 4)
            break; // 文件结束

        if (dim <= 0)
            throw runtime_error("无效维度: " + to_string(dim));

        // 读取数据块 (小端int32数组)
        vector<int32_t> vec(dim);
        in.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(int32_t));
        if (in.gcount() != dim * sizeof(int32_t))
            throw runtime_error("文件数据不完整");

        data.push_back(move(vec));
    }

    return data;
}

vector<vector<float>> read_fvecs(const string &filename)
{
    ifstream in(filename, ios::binary);
    if (!in)
        throw runtime_error("无法打开文件: " + filename);

    vector<vector<float>> data;

    while (true)
    {
        // 读取维度 (小端int32)
        int32_t dim;
        in.read(reinterpret_cast<char *>(&dim), 4);
        if (in.gcount() != 4)
            break; // 文件结束

        if (dim <= 0)
            throw runtime_error("无效维度: " + to_string(dim));

        // 读取数据块
        vector<float> vec(dim);
        in.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
        if (in.gcount() != dim * sizeof(float))
            throw runtime_error("文件数据不完整");

        data.push_back(move(vec));
    }

    return data;
}

float *read_fvecs(const string &filename, size_t &n, size_t &d)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open file: " + filename);

    // Read first dimension (little-endian int32)
    int32_t dim;
    in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
    if (in.gcount() != sizeof(dim))
        throw std::runtime_error("Failed to read dimension from file: " + filename);
    if (dim <= 0)
        throw std::runtime_error("Invalid dimension: " + std::to_string(dim));
    d = static_cast<size_t>(dim);

    // Determine file size
    auto cur_pos = in.tellg();
    in.seekg(0, std::ios::end);
    auto file_size = in.tellg();
    in.seekg(cur_pos);

    const size_t header_bytes = sizeof(int32_t);
    const size_t vector_bytes = d * sizeof(float);
    const size_t record_size = header_bytes + vector_bytes;

    if ((file_size % record_size) != 0)
        throw std::runtime_error("File size is not a multiple of record size");

    n = static_cast<size_t>(file_size / record_size);

    // Allocate contiguous memory for all vectors
    float *data = new float[n * d];

    // Rewind to beginning and load
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; ++i)
    {
        int32_t cur_dim;
        in.read(reinterpret_cast<char *>(&cur_dim), sizeof(cur_dim));
        if (in.gcount() != sizeof(cur_dim))
            throw std::runtime_error("Unexpected EOF reading dimension at index " + std::to_string(i));
        if (cur_dim != dim)
            throw std::runtime_error("Inconsistent dimension at index " + std::to_string(i));

        // Read vector directly into contiguous block
        float *vec_ptr = data + i * d;
        in.read(reinterpret_cast<char *>(vec_ptr), vector_bytes);
        if (static_cast<size_t>(in.gcount()) != vector_bytes)
            throw std::runtime_error("Unexpected EOF reading vector at index " + std::to_string(i));
    }

    return data;
}

vector<vector<uint32_t>> read_uivecs(const string &filename) // 读取 32 位无符号整数
{
    ifstream in(filename, ios::binary);
    if (!in)
        throw runtime_error("无法打开文件: " + filename);

    vector<vector<uint32_t>> data;
    int dim;

    while (in.read(reinterpret_cast<char *>(&dim), sizeof(int)))
    {
        if (dim == 0)
            throw runtime_error("无效维度: 0");

        vector<uint32_t> vec(dim);
        in.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(uint32_t));
        if (in.gcount() != dim * sizeof(uint32_t))
            throw runtime_error("文件数据不完整");

        data.push_back(move(vec));
    }

    return data;
}

double compute_mrr_at_r(const std::vector<std::vector<uint32_t>> &labels,
                   const std::vector<std::vector<size_t>> &query_res,
                   size_t R)
{
    // 校验输入维度
    const size_t batch_size = labels.size();
    if (query_res.size() != batch_size)
    {
        throw std::invalid_argument("Labels and query results batch size mismatch");
    }

    if (R == 0)
    {
        throw std::invalid_argument("R must be greater than 0");
    }

    double total_mrr = 0.0;
    size_t valid_queries = 0;

    for (size_t i = 0; i < batch_size; ++i)
    {
        // 获取当前 query 的所有 groundtruth
        const auto &current_labels = labels[i];

        // 跳过没有 groundtruth 的无效 query
        if (current_labels.empty())
        {
            continue;
        }

        // 构建 groundtruth 集合
        std::unordered_set<uint32_t> gt_set(current_labels.begin(), current_labels.end());

        // 遍历前 R 个结果寻找首次命中
        bool found = false;
        size_t first_rank = 0;
        const size_t actual_R = std::min(R, query_res[i].size());

        for (size_t j = 0; j < actual_R; ++j)
        {
            if (gt_set.count(query_res[i][j]))
            {
                first_rank = j + 1; // 排名从1开始
                found = true;
                break;
            }
        }

        // 累加倒数排名
        if (found)
        {
            total_mrr += 1.0 / first_rank;
        } // 未命中时默认加0

        ++valid_queries;
    }

    if (valid_queries == 0)
    {
        throw std::runtime_error("No valid queries with groundtruth");
    }

    return total_mrr / valid_queries;
}

double compute_ndcg_at_r(const std::vector<std::vector<uint32_t>> &labels,
                    const std::vector<std::vector<size_t>> &query_res,
                    size_t R,
                    const std::vector<std::vector<uint32_t>> &relevance_scores)
{
    // 校验输入维度
    const size_t batch_size = labels.size();
    if (query_res.size() != batch_size)
    {
        throw std::invalid_argument("Labels and query results batch size mismatch");
    }

    bool use_multilevel = !relevance_scores.empty();
    if (use_multilevel && relevance_scores.size() != batch_size)
    {
        throw std::invalid_argument("relevance_scores must have same batch size as labels");
    }

    if (R == 0)
    {
        throw std::invalid_argument("R must be greater than 0");
    }

    double total_ndcg = 0.0;
    size_t valid_queries = 0;

    for (size_t i = 0; i < batch_size; ++i)
    {
        const auto &current_labels = labels[i];
        if (current_labels.empty())
        {
            continue; // 跳过无groundtruth的查询
        }

        // 校验相关性分数维度
        if (use_multilevel && relevance_scores[i].size() != current_labels.size())
        {
            throw std::invalid_argument("relevance_scores[" + std::to_string(i) + "] size mismatch");
        }

        // 构建标签到分数的映射
        std::unordered_map<uint32_t, int> label_score_map;
        std::vector<int> gt_scores;
        for (size_t j = 0; j < current_labels.size(); ++j)
        {
            int score = use_multilevel ? relevance_scores[i][j] : 1;
            label_score_map[current_labels[j]] = score;
            gt_scores.push_back(score);
        }

        // 计算 DCG@R
        double dcg = 0.0;
        const size_t actual_R = std::min(R, query_res[i].size());
        for (size_t j = 0; j < actual_R; ++j)
        {
            const auto &item = query_res[i][j];
            auto it = label_score_map.find(item);
            if (it != label_score_map.end())
            {
                dcg += it->second / std::log2(j + 2); // rank从1开始计算
            }
        }

        // 计算 IDCG@R
        std::sort(gt_scores.rbegin(), gt_scores.rend()); // 降序排序
        std::vector<int> ideal_scores(gt_scores.begin(), gt_scores.begin() + std::min(R, gt_scores.size()));
        ideal_scores.resize(R, 0); // 填充0直到R个元素

        double idcg = 0.0;
        for (size_t j = 0; j < R; ++j)
        {
            idcg += ideal_scores[j] / std::log2(j + 2);
        }

        // 计算nDCG
        if (idcg > 0)
        {
            total_ndcg += dcg / idcg;
        } // 否则保持加0

        ++valid_queries;
    }

    if (valid_queries == 0)
    {
        throw std::runtime_error("No valid queries with groundtruth");
    }

    return total_ndcg / valid_queries;
}


std::vector<size_t> select_random_indices(size_t _n_base_data, size_t k)
{
    std::vector<size_t> indices;

    // 处理无效输入
    if (k == 0 || _n_base_data == 0 || k > _n_base_data)
    {
        return indices; // 返回空vector
    }

    // 生成索引列表 0, 1, 2, ..., _n_base_data-1
    indices.reserve(_n_base_data);
    for (size_t i = 0; i < _n_base_data; ++i)
    {
        indices.push_back(i);
    }

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());

    // 打乱列表
    std::shuffle(indices.begin(), indices.end(), gen);

    // 截取前k个元素
    indices.resize(k);

    return indices;
}

void l2_normalize_rows(std::vector<std::vector<float>> &data)
{
    for (auto &row : data)
    {
        // 计算当前行的 L2 范数
        float sum_sq = 0.0f;
        for (float v : row)
        {
            sum_sq += v * v;
        }
        float norm = std::sqrt(sum_sq);
        // 防止除 0，若范数为 0 则跳过
        if (norm > 0.0f)
        {
            for (auto &v : row)
            {
                v /= norm;
            }
        }
    }
}


bool fileExists(const std::string &filename)
{
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}
