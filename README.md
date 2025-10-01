# Isolation Kernel Meets LLMs: Lightweight, Training-free Binary Embeddings for Fast Retrieval

## Hardware Requirements

IKE is developed using the AVX512 instruction set. Before running the code, please ensure your CPU supports the following flags: `avx512f`, `avx512cd`, `avx512vl`, `avx512dq`, `avx512bw`, `avx512_vpopcntdq`. For Linux operating systems, you can check using the following command:

```bash
lscpu | grep avx512
```

## Building from source

IKE is developed in C++ and can be built from source using CMake. The [test](./test/) folder contains three test programs: [Exhaustive_Search_with_IKE.cpp](./test/Exhaustive_Search_with_IKE.cpp), [HNSW_with_IKE.cpp](./test/HNSW_with_IKE.cpp), and [IVF_with_IKE.cpp](./test/IVF_with_IKE.cpp). Please follow the steps below to compile:

### Step 1: Download the Source Code and Create a Build Directory

``` bash
git clone https://github.com/Zed-Zed-b/IKE
cd IKE
mkdir build && cd build
```

### Step 2: Configure with CMake

Part of IKE's implementation is based on the [Faiss library](https://github.com/facebookresearch/faiss) v1.9.0. Therefore, when running CMake, you need to specify the include path and the dynamic library path for Faiss. The simplest method is to create a new [conda](https://docs.conda.io/en/latest/) environment and install version 1.9.0 of the `faiss-cpu` package in this environment by following the instructions at [this document](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md). After activating this conda environment, run the following CMake command:

``` bash
cmake ..  -DFAISS_INCLUDE_ROOT=$CONDA_PREFIX/include -DFAISS_LIB_ROOT=$CONDA_PREFIX/lib
```

Alternatively, you can choose to compile and install the Faiss library manually from source and specify the corresponding paths using the `-DFAISS_INCLUDE_ROOT` and `-DFAISS_LIB_ROOT` options.

### Step 3: Compile with Make

```
make -j32
```

After completion, you will find three executable files in the `build` directory: `Exhaustive_Search_with_IKE`, `HNSW_with_IKE`, and `IVF_with_IKE`.

## Running the Programs

### Preparing the Dataset

We provide two pre-computed embedding datasets: [Istella22](https://istella.ai/datasets/istella22-dataset/) ([Qwen3-embedding](https://github.com/QwenLM/Qwen3-Embedding)) and [FEVER-HN](https://huggingface.co/datasets/mteb/FEVER_test_top_250_only_w_correct-v2) ([LLM2Vec](https://github.com/McGill-NLP/llm2vec)) [**here**](https://drive.google.com/drive/folders/1sFM-w9QxAkT_5MR7Tt00UwyHQUnHed1R?usp=sharing). Please download and extract the datasets (please ensure you are in the project's main directory, i.e., the `IKE` folder):

``` bash
mkdir data && cd data

# Download two datasets through shared URL

# Unzip
unzip feverHN_l2v.zip 
unzip istella22_qwen3.zip

# Remove zip files
rm feverHN_l2v.zip istella22_qwen3.zip

# Return to project root
cd ..
```

Now you can run the experiments! The following examples use the Istella22 (Qwen3-embedding) dataset.

**Note**: Since Istella22 is a multi-level relevance ranking retrieval dataset, you need to specify the path to the ground truth score file using the `-score` option to correctly calculate the retrieval metrics. The FEVER-HN dataset uses binary scores, so this option is not required.

### Exhaustive Search with IKE

Run the following command:

``` bash
./build/Exhaustive_Search_with_IKE -t 4096 --psi 2 -corpus "./data/istella22_qwen3/istella22_base_4096_qwen3.fvecs" -query "./data/istella22_qwen3/istella22_query_4096_qwen3.fvecs" -label "./data/istella22_qwen3/istella22_groundtruth.uivecs" -score "./data/istella22_qwen3/istella22_gt_scores.uivecs"
```

### HNSW with IKE

Run the following command:

``` bash
./build/HNSW_with_IKE -t 4096 --psi 2 -corpus "./data/istella22_qwen3/istella22_base_4096_qwen3.fvecs" -query "./data/istella22_qwen3/istella22_query_4096_qwen3.fvecs" -label "./data/istella22_qwen3/istella22_groundtruth.uivecs" -score "./data/istella22_qwen3/istella22_gt_scores.uivecs"
```

### IVF with IKE

Run the following command:

``` bash
./build/IVF_with_IKE -t 4096 --psi 2 -corpus "./data/istella22_qwen3/istella22_base_4096_qwen3.fvecs" -query "./data/istella22_qwen3/istella22_query_4096_qwen3.fvecs" -label "./data/istella22_qwen3/istella22_groundtruth.uivecs" -score "./data/istella22_qwen3/istella22_gt_scores.uivecs"
```
