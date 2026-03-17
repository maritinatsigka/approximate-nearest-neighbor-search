#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include "cli.hpp"
#include "dataset.hpp"
#include "hypercube.hpp"
#include "LSH.hpp"
#include "ivfflat.hpp"
#include "utils.hpp"
#include "ivfpq.hpp"

using Clock = std::chrono::high_resolution_clock;


//writes a single query result block to the output file:
//prints the query id
//prints the top-N approximate neighbor ids and their approximate distances
//prints the corresponding ground-truth distances for comparison
void write_query_block(std::ofstream& out,int qid, const std::vector<double>& q, const std::vector<std::vector<double>>& data,const std::vector<int>& approxIds, const std::vector<int>& truthIds){
    out << "Query: " << qid << "\n";
    out << std::fixed << std::setprecision(6);

    const int N = (int)approxIds.size();
    for (int i = 0; i < N; ++i){
        const int aid = approxIds[i];
        out << "Nearest neighbor-" << (i+1) << ": " << aid << "\n";
        out << "distanceApproximate: " << l2(q, data[aid]) << "\n";

        int ti = truthIds.empty() ? -1 : std::min(i, (int)truthIds.size() - 1);
        double dTrue = (ti < 0) ? l2(q, data[aid]) : l2(q, data[truthIds[ti]]);
        out << "distanceTrue: " << dTrue << "\n";
    }
    out << "\n";
}

int main(int argc, char** argv){
    CLI cli(argc, argv);

    // Required args: -d -q -type -o and exactly one of -lsh / -hypercube / -ivfflat / -ivfpq
    if (!cli.has("-d") || !cli.has("-q") || !cli.has("-type") || !cli.has("-o")) {
        std::cerr << "Usage:\n"
                  << "  ./search -d <data> -q <queries> -type <mnist|sift> -o <outfile>\n"
                  << "    [-lsh -k <int> -L <int> -w <double>]\n"
                  << "    [-hypercube -kproj <int> -M <int> -probes <int> -w <double>]\n"
                  << "    [-ivfflat -kclusters <int> -nprobe <int>]\n"
                  << "    [-ivfpq -kclusters <int> -nprobe <int> -M <int> -nbits <int>]\n"
                  << "    [-N <int>] [-R <double>] [-range true|false] [-seed <int>]\n";
        return 1;
    }

    const std::string dpath = cli.get("-d");
    const std::string qpath = cli.get("-q");
    const std::string outpath = cli.get("-o");
    const bool use_hc  = cli.has("-hypercube");
    const bool use_lsh = cli.has("-lsh");
    const bool use_ivf = cli.has("-ivfflat");
    const bool use_ivfpq = cli.has("-ivfpq");

    if ((use_hc?1:0) + (use_lsh?1:0) + (use_ivf?1:0) + (use_ivfpq?1:0) != 1){
        std::cerr << "Error: choose exactly one of -hypercube, -lsh, -ivfflat, -ivfpq\n";
        return 1;
    }

    //dataset type handling
    const std::string typeStr = cli.get("-type");
    DataType dtype;
    if (typeStr == "mnist") dtype = DataType::MNIST;
    else if (typeStr == "sift") dtype = DataType::SIFT;
    else{
        std::cerr << "Error: -type must be exactly 'mnist' or 'sift'\n";
        return 1;
    }

    //common params.
    unsigned seed = (unsigned)cli.geti("-seed", 1);
    bool do_range = (cli.get("-range", "false") == "true");
    double w = cli.getd("-w", 4.0);
    int N = cli.geti("-N", 1);
    double Rdef = (dtype == DataType::MNIST ? 2000.0 : 2.0);
    double R = cli.getd("-R", Rdef);

    //load base set and queries
    std::vector<std::vector<double>> data, queries;
    if (!load_dataset(dpath, dtype, data) || !load_dataset(qpath, dtype, queries)){
        std::cerr << "Dataset loading failed.\n";
        return 2;
    }
    if (data.empty() || queries.empty()){
        std::cerr << "Empty dataset or query set.\n";
        return 2;
    }
    int n = (int)data.size(), dim = (int)data[0].size();

    //open output file
    std::ofstream out(outpath);
    if (!out.is_open()){
        std::cerr << "Could not open output file: " << outpath << "\n";
        return 3;
    }

    //metrics accumulators.
    double sumAF = 0.0, sumRecall = 0.0, sumApproxMs = 0.0, sumTrueMs = 0.0;

    //helper or lamba functions.
    auto bestDist = [&](const std::vector<int>& ids, const std::vector<double>& q)->double{
        if (ids.empty()) return std::numeric_limits<double>::infinity();
        double best = std::numeric_limits<double>::infinity();
        for (int id : ids) best = std::min(best, l2(q, data[id]));
        return best;
    };
    auto overlap_count = [&](const std::vector<int>& A, const std::vector<int>& B)->int{
        int cnt = 0;
        for (int a : A){
            for (int b : B){ 
                if (a == b){ 
                    ++cnt; 
                    break; 
                } 
            }
        }
        return cnt;
    };

    //HYPERCUBE
    if (use_hc){
        int kproj = cli.geti("-kproj", 14);
        int M = cli.geti("-M", 10);
        int probes = cli.geti("-probes", 2);

        Hypercube hc(n, dim, kproj, w, M, probes, seed);
        hc.BuildIndex(data);

        out << "Hypercube\n\n";

        //per-query evaluation loop
        for (int qi = 0; qi < (int)queries.size(); ++qi){
            const auto& q = queries[qi];

            auto tA0 = Clock::now();
            std::vector<int> approx = hc.ANN(q, N);   //approximate top-N
            auto tA1 = Clock::now();

            auto tT0 = Clock::now();
            std::vector<int> truth = brute_force_topN(data, q, N);   //ground truth
            auto tT1 = Clock::now();

            double dApprox = bestDist(approx, q);
            double dTrue = bestDist(truth, q);
            double AF = (dTrue==0.0? 1.0 : dApprox/dTrue);  //approximation factor
            int overlap = overlap_count(approx, truth);
            double recall = truth.empty()? 0.0 : (double)overlap / (double)N;

            sumAF += AF;
            sumRecall += recall;
            sumApproxMs += std::chrono::duration_cast<std::chrono::microseconds>(tA1 - tA0).count()/1000.0;
            sumTrueMs += std::chrono::duration_cast<std::chrono::microseconds>(tT1 - tT0).count()/1000.0;

            write_query_block(out, qi, q, data, approx, truth);

            //optional range search using the index's Range method
            if (do_range){
                std::vector<int> rnears = hc.Range(q, R);
                out << "R-near neighbors:\n";
                for (int id : rnears) out << id << "\n";
                out << "\n";
            }
        }

        //aggregate statistics over all queries
        size_t Q = queries.size();
        out << std::fixed << std::setprecision(6);
        out << "Average AF: " << (Q? (sumAF/Q) : 0.0) << "\n";
        out << "Recall@N: " << (Q? (sumRecall/Q) : 0.0) << "\n";
        double QPS = (sumApproxMs>0.0 && Q) ? (1000.0 * (double)Q / sumApproxMs) : 0.0;
        out << "QPS: " << QPS << "\n";
        out << "tApproximateAverage: " << (Q? (sumApproxMs/Q) : 0.0) << "\n";
        out << "tTrueAverage: " << (Q? (sumTrueMs/Q) : 0.0) << "\n";
        return 0;
    }
    //LSH
    if (use_lsh){
        int k = cli.geti("-k", 4);  //hashes per table
        int L = cli.geti("-L", 5);  //number of hash tables

        LSH lsh(dim, k, L, w, seed);
        lsh.Build_index(data);
        out << "LSH\n\n";
        for (int qi = 0; qi < (int)queries.size(); ++qi){
            const auto& q = queries[qi];

            auto tA0 = Clock::now();
            std::vector<int> approx = lsh.ANN(q, N);
            auto tA1 = Clock::now();

            auto tT0 = Clock::now();
            std::vector<int> truth = brute_force_topN(data, q, N);
            auto tT1 = Clock::now();

            double dApprox = bestDist(approx, q);
            double dTrue = bestDist(truth, q);
            double AF = (dTrue==0.0? 1.0 : dApprox/dTrue);
            int overlap = overlap_count(approx, truth);
            double recall = truth.empty()? 0.0 : (double)overlap / (double)N;

            sumAF += AF;
            sumRecall += recall;
            sumApproxMs += std::chrono::duration_cast<std::chrono::microseconds>(tA1 - tA0).count()/1000.0;
            sumTrueMs += std::chrono::duration_cast<std::chrono::microseconds>(tT1 - tT0).count()/1000.0;

            write_query_block(out, qi, q, data, approx, truth);

            //for LSH, range is done exactly via brute-force helper
            if (do_range){
                // exact range per brief label
                std::vector<int> rnears = range_search(data, q, R);
                out << "R-near neighbors:\n";
                for (int id : rnears) out << id << "\n";
                out << "\n";
            }
        }

        size_t Q = queries.size();
        out << std::fixed << std::setprecision(6);
        out << "Average AF: " << (Q? (sumAF/Q) : 0.0) << "\n";
        out << "Recall@N: " << (Q? (sumRecall/Q) : 0.0) << "\n";
        double QPS = (sumApproxMs>0.0 && Q) ? (1000.0 * (double)Q / sumApproxMs) : 0.0;
        out << "QPS: " << QPS << "\n";
        out << "tApproximateAverage: " << (Q? (sumApproxMs/Q) : 0.0) << "\n";
        out << "tTrueAverage: " << (Q? (sumTrueMs/Q) : 0.0) << "\n";
        return 0;
    }

    //IVFFlat
    if (use_ivf){
        int kclusters = cli.geti("-kclusters", 50);  //number of coarse centroids (lists)
        int nprobe = cli.geti("-nprobe", 5);    //number of lists to probe per query

        IVFFlat ivf(n, dim, kclusters, nprobe, seed);
        ivf.Build_index(data);

        out << "IVFFlat\n\n";

        for (int qi = 0; qi < (int)queries.size(); ++qi){
            const auto& q = queries[qi];

            auto tA0 = Clock::now();
            std::vector<int> approx = ivf.ANN(q, N);
            auto tA1 = Clock::now();

            auto tT0 = Clock::now();
            std::vector<int> truth = brute_force_topN(data, q, N);
            auto tT1 = Clock::now();

            double dApprox = bestDist(approx, q);
            double dTrue = bestDist(truth, q);
            double AF = (dTrue==0.0? 1.0 : dApprox/dTrue);
            int overlap = overlap_count(approx, truth);
            double recall = truth.empty()? 0.0 : (double)overlap / (double)N;

            sumAF += AF;
            sumRecall += recall;
            sumApproxMs += std::chrono::duration_cast<std::chrono::microseconds>(tA1 - tA0).count()/1000.0;
            sumTrueMs += std::chrono::duration_cast<std::chrono::microseconds>(tT1 - tT0).count()/1000.0;

            write_query_block(out, qi, q, data, approx, truth);

            //IVF range using index's Range method
            if (do_range){
                std::vector<int> rnears = ivf.Range(q, R);
                out << "R-near neighbors:\n";
                for (int id : rnears) out << id << "\n";
                out << "\n";
            }
        }

        size_t Q = queries.size();
        out << std::fixed << std::setprecision(6);
        out << "Average AF: " << (Q? (sumAF/Q) : 0.0) << "\n";
        out << "Recall@N: " << (Q? (sumRecall/Q) : 0.0) << "\n";
        double QPS = (sumApproxMs>0.0 && Q) ? (1000.0 * (double)Q / sumApproxMs) : 0.0;
        out << "QPS: " << QPS << "\n";
        out << "tApproximateAverage: " << (Q? (sumApproxMs/Q) : 0.0) << "\n";
        out << "tTrueAverage: " << (Q? (sumTrueMs/Q) : 0.0) << "\n";
        return 0;
    }

    if (use_ivfpq){
        int kclusters = cli.geti("-kclusters", 50);
        int nprobe = cli.geti("-nprobe", 5);
        int M = cli.geti("-M", 16);   // number of PQ subspaces
        int nbits = cli.geti("-nbits", 8);    //bits per subspace code (Ks=2^nbits)

        IVFPQ ivfpq(n, dim, kclusters, nprobe, M, nbits, seed);
        ivfpq.Build_index(data);

        out << "IVFPQ\n\n";

        for (int qi = 0; qi < (int)queries.size(); ++qi){
            const auto& q = queries[qi];

            auto tA0 = Clock::now();
            std::vector<int> approx = ivfpq.ANN(q, N);
            auto tA1 = Clock::now();

            auto tT0 = Clock::now();
            std::vector<int> truth = brute_force_topN(data, q, N);
            auto tT1 = Clock::now();

            double dApprox = bestDist(approx, q);
            double dTrue = bestDist(truth, q);
            double AF = (dTrue==0.0? 1.0 : dApprox/dTrue);
            int overlap = overlap_count(approx, truth);
            double recall = truth.empty()? 0.0 : (double)overlap / (double)N;

            sumAF += AF;
            sumRecall += recall;
            sumApproxMs += std::chrono::duration_cast<std::chrono::microseconds>(tA1 - tA0).count()/1000.0;
            sumTrueMs += std::chrono::duration_cast<std::chrono::microseconds>(tT1 - tT0).count()/1000.0;

            write_query_block(out, qi, q, data, approx, truth);

            //IVFPQ range using full-precision check on probed lists
            if (do_range){
                std::vector<int> rnears = ivfpq.Range(q, R);
                out << "R-near neighbors:\n";
                for (int id : rnears) out << id << "\n";
                out << "\n";
            }
        }

        size_t Q = queries.size();
        out << std::fixed << std::setprecision(6);
        out << "Average AF: " << (Q? (sumAF/Q) : 0.0) << "\n";
        out << "Recall@N: " << (Q? (sumRecall/Q) : 0.0) << "\n";
        double QPS = (sumApproxMs>0.0 && Q) ? (1000.0 * (double)Q / sumApproxMs) : 0.0;
        out << "QPS: " << QPS << "\n";
        out << "tApproximateAverage: " << (Q? (sumApproxMs/Q) : 0.0) << "\n";
        out << "tTrueAverage: " << (Q? (sumTrueMs/Q) : 0.0) << "\n";
        return 0;
    }

    return 0;
}