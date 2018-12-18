#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <random>
#include <omp.h>
#include <cassert>
#include <cstring>
#include <iterator>


using namespace std;

// initializes the random number generator
static default_random_engine GLOBAL_GENERATOR;
// creates a uniform distr. generator for number between 0->1
static uniform_real_distribution<double> UNIFORM(0, 1);

//              s   r    o
typedef tuple<int, int, int> triplet;

vector<string> read_first_column(const string& fname) {
    // reads from a column of a file
    ifstream ifs(fname, ios::in); // file to read
    string line; // line holder variable
    string item; // temp variable for current line read/parsed
    vector<string> items; // list of all itmes read

    assert(!ifs.fail()); // makes sure file open

    while (getline(ifs, line)) { // goes through all line of file
        stringstream ss(line); // makes line in string stream
        ss >> item; // extract the item from the current line
        items.push_back(item); // stores item for returning
    }
    ifs.close(); // closes the file
    return items; // returns list of items
}

unordered_map<string, int> create_id_mapping(const vector<string>& items) {
    // creates a mapping between items and unique IDs
    unordered_map<string, int> map;
    for (int i = 0; i < (int) items.size(); i++)
        map[items[i]] = i;

    return map;
}

vector<triplet> create_sros(
        const string& fname,
        const unordered_map<string, int>& ent_map,
        const unordered_map<string, int>& rel_map) {
    // creates a dataset of s,r,o triplets
    ifstream ifs(fname, ios::in); // file of triplets
    string line; // triplet line variable
    string s, r, o; // subj, obj, rel holders
    vector<triplet> sros; // list of s,r,o triplets to return
    // make sure dataset file is open
    assert(!ifs.fail());
    while (getline(ifs, line)) { // go through all lines in dataset
        stringstream ss(line); //make stringstream of line
        ss >> s >> r >> o; // assign the subj, obj, rel to s,o,r
        // add triplet to list while mapping names to unique IDs
        sros.push_back( make_tuple(ent_map.at(s), rel_map.at(r), ent_map.at(o)) );
    }
    ifs.close(); // close file
    return sros; // return triplets list
}

vector<vector<double>> uniform_matrix(int m, int n, double l, double h) {
    // creates an MxN matrix with random numbers uniformly distributed between
    // h and l
    vector<vector<double>> matrix;
    matrix.resize(m); // creates M dim
    for (int i = 0; i < m; i++)
        matrix[i].resize(n); // creates N dim

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            // populates the matrix with random numbers uniformly b/w h and l
            matrix[i][j] = (h-l)*UNIFORM(GLOBAL_GENERATOR) + l;

    return matrix;
}

vector<vector<double>> const_matrix(int m, int n, double c) {
    // creates an MxN constant matrix initialized to c
    vector<vector<double>> matrix;
    matrix.resize(m);
    for (int i = 0; i < m; i++)
        matrix[i].resize(n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            matrix[i][j] = c;

    return matrix;
}

vector<int> range(int n) {
    // creates a vector of size n, with elements [0 ... n-1]
    vector<int> v;
    v.reserve(n);
    for (int i = 0; i < n; i++)
        v.push_back(i);
    return v;
}

// XXX: EXTRA FUNCTION NOT USED???
void l2_normalize(vector<double>& vec) {
    double sq_norm = 0;
    for (unsigned i = 0; i < vec.size(); i++)
        sq_norm += vec[i] * vec[i];
    double norm = sqrt(sq_norm);
    for (unsigned i = 0; i < vec.size(); i++)
        vec[i] /= norm;
}

double sigmoid(double x, double cutoff=30) {
    // does a sigmoid x
    if (x > +cutoff) return 1.;
    if (x < -cutoff) return 0.;
    return 1./(1.+exp(-x));
}

class SROBucket {
    // class that represents all s,r,o in train, test, and valid sets
    unordered_set<int64_t> __sros; // list of all s,r,o hashes
    unordered_map<int64_t, vector<int>> __sr2o; // mapping from s,r hash to o
    unordered_map<int64_t, vector<int>> __or2s; // mapping from r,o hash to s
    
    // hashes an s,r,o triplet
    int64_t hash(int a, int b, int c) const {
        int64_t x = a;
        x = (x << 20) + b;
        return (x << 20) + c;
    }
    // hashes an s,r or o,r pair
    int64_t hash(int a, int b) const {
        int64_t x = a;
        return (x << 32) + b;
    }

public:
    SROBucket(const vector<triplet>& sros) {
        // takes the SROs all into 1 bucket
        for (auto sro : sros) {
            // extracts the s,r,o
            int s = get<0>(sro);
            int r = get<1>(sro);
            int o = get<2>(sro);
            // adds s,r,o hash to s,r,o hash list
            int64_t __sro = hash(s, r, o);
            __sros.insert(__sro);
            // adds s,r hash to mapping and puts o as output
            int64_t __sr = hash(s, r);
            if (__sr2o.find(__sr) == __sr2o.end())
                __sr2o[__sr] = vector<int>();
            __sr2o[__sr].push_back(o);
            // adds r,o hash to mapping and puts s as output
            int64_t __or = hash(o, r);
            if (__or2s.find(__or) == __or2s.end())
                __or2s[__or] = vector<int>();
            __or2s[__or].push_back(s);
        }
    }

    bool contains(int a, int b, int c) const {
        // checks bucket for some s,r,o
        return __sros.find( hash(a, b, c) ) != __sros.end();
    }

    vector<int> sr2o(int s, int r) const {
        // maps an s,r has to an o
        return __sr2o.at(hash(s,r));
    }

    vector<int> or2s(int o, int r) const {
        // maps an r,o has to an s
        return __or2s.at(hash(o,r));
    }
};

// try sample pairs
class NegativeSampler {
    uniform_int_distribution<int> unif_e; // random entity generator
    uniform_int_distribution<int> unif_r; // random relation generator
    default_random_engine generator; // random seed for generators

public:
    NegativeSampler(int ne, int nr, int seed) :
        // creates a random num generator in range of entities
        // creates a random num generator in range of relations
        unif_e(0, ne-1), unif_r(0, nr-1), generator(seed) {}

    int random_entity() {
        // returns and random entity number
        return unif_e(generator);
    }
    
    int random_relation() {
        // returns a random relation number
        return unif_r(generator);
    }
};

class Model {

protected:
    double eta;
    double gamma;
    const double init_b = 1e-2;
    const double init_e = 1e-6;

    vector<vector<double>> E; // entity embeddings
    vector<vector<double>> R; // relation embeddings
    vector<vector<double>> E_g; // entity matrix for adagrad
    vector<vector<double>> R_g; // relation matrix for adagrad

public:

    Model(double eta, double gamma) {
        this->eta = eta; // related to learning rate in adagrad
        this->gamma = gamma; // related to training gradient
    }

    void save(const string& fname) {
        ofstream ofs(fname, ios::out);

        for (unsigned i = 0; i < E.size(); i++) {
            for (unsigned j = 0; j < E[i].size(); j++)
                ofs << E[i][j] << ' ';
            ofs << endl;
        }

        for (unsigned i = 0; i < R.size(); i++) {
            for (unsigned j = 0; j < R[i].size(); j++)
                ofs << R[i][j] << ' ';
            ofs << endl;
        }

        ofs.close();
    }

    void load(const string& fname) {
        // loads a KGE model from the file fname as ifstream
        ifstream ifs(fname, ios::in);

        for (unsigned i = 0; i < E.size(); i++)
            for (unsigned j = 0; j < E[i].size(); j++)
                // extracts entity embeddings
                ifs >> E[i][j];

        for (unsigned i = 0; i < R.size(); i++)
            for (unsigned j = 0; j < R[i].size(); j++)
                // extracts the relation embeddings
                ifs >> R[i][j];
        // closes model file
        ifs.close();
    }

    void adagrad_update(
            // adjusts the learning rate according to adagrad
            int s,
            int r,
            int o,
            const vector<double>& d_s,
            const vector<double>& d_r,
            const vector<double>& d_o) {

        for (unsigned i = 0; i < E[s].size(); i++) E_g[s][i] += d_s[i] * d_s[i];
        for (unsigned i = 0; i < R[r].size(); i++) R_g[r][i] += d_r[i] * d_r[i];
        for (unsigned i = 0; i < E[o].size(); i++) E_g[o][i] += d_o[i] * d_o[i];

        for (unsigned i = 0; i < E[s].size(); i++) E[s][i] -= eta * d_s[i] / sqrt(E_g[s][i]);
        for (unsigned i = 0; i < R[r].size(); i++) R[r][i] -= eta * d_r[i] / sqrt(R_g[r][i]);
        for (unsigned i = 0; i < E[o].size(); i++) E[o][i] -= eta * d_o[i] / sqrt(E_g[o][i]);
    }

    void train(int s, int r, int o, bool is_positive) {
        vector<double> d_s;
        vector<double> d_r;
        vector<double> d_o;

        d_s.resize(E[s].size());
        d_r.resize(R[r].size());
        d_o.resize(E[o].size());

        double offset = is_positive ? 1 : 0;
        // loss for label 1 or 0, and score of s,r,o triple
        double d_loss = sigmoid(score(s, r, o)) - offset;
        // gradients for s,r,o triple
        score_grad(s, r, o, d_s, d_r, d_o);
        // backprop s,r,o triple
        for (unsigned i = 0; i < d_s.size(); i++) d_s[i] *= d_loss;
        for (unsigned i = 0; i < d_r.size(); i++) d_r[i] *= d_loss;
        for (unsigned i = 0; i < d_o.size(); i++) d_o[i] *= d_loss;
        // adagrad updates below (weight decay included)
        double gamma_s = gamma / d_s.size();
        double gamma_r = gamma / d_r.size();
        double gamma_o = gamma / d_o.size();
        for (unsigned i = 0; i < d_s.size(); i++) d_s[i] += gamma_s * E[s][i];
        for (unsigned i = 0; i < d_r.size(); i++) d_r[i] += gamma_r * R[r][i];
        for (unsigned i = 0; i < d_o.size(); i++) d_o[i] += gamma_o * E[o][i];
        adagrad_update(s, r, o, d_s, d_r, d_o);
    }

    virtual double score(int s, int r, int o) const = 0;

    virtual void score_grad(
            int s,
            int r,
            int o,
            vector<double>& d_s, 
            vector<double>& d_r, 
            vector<double>& d_o) {};
};

class Evaluator {
    int ne;
    int nr;
    const vector<triplet>& sros;
    const SROBucket& sro_bucket;

public:
    Evaluator(int ne, int nr, const vector<triplet>& sros, const SROBucket& sro_bucket) :
        // sets the number of relations and entities, also gives the set of 
        // s,r,o used to either test, train, or validate and the complete s,r,o
        // bucket        
        ne(ne), nr(nr), sros(sros), sro_bucket(sro_bucket) {}
        
    /*void fake_eval(const Model *model) {
        int N = this->sros.size();
        vector<double> scores;
        for (int i = 0; i < N; i++) {
            const triplet& sro = sros[i];
            int s = get<0>(sro);
            int r = get<1>(sro);
            int o = get<2>(sro);
            double base_score = sigmoid(model->score(s, r, o) - 1e-32);
            scores.push_back(base_score);
        }
        FILE* f1 = fopen("scores.txt","w");
        for (int i=0; i<N; i++) {
            fprintf(f1,"%.9lf\n",scores[i]);
        }
    }*/

    unordered_map<string, double> evaluate(const Model *model, int truncate) {
        // complete training set size
        int N = this->sros.size();
        
        // sets the batch size to N
        if (truncate > 0)
            N = min(N, truncate);

        double mrr_s = 0.;
        double mrr_r = 0.;
        double mrr_o = 0.;

        double mrr_s_raw = 0.;
        double mrr_o_raw = 0.;

        double mr_s = 0.;
        double mr_r = 0.;
        double mr_o = 0.;

        double mr_s_raw = 0.;
        double mr_o_raw = 0.;

        double hits01_s = 0.; 
        double hits01_r = 0.;
        double hits01_o = 0.;

        double hits03_s = 0.;
        double hits03_r = 0.;
        double hits03_o = 0.;

        double hits10_s = 0.;
        double hits10_r = 0.;
        double hits10_o = 0.;

        #pragma omp parallel for reduction(+: mrr_s, mrr_r, mrr_o, mr_s, mr_r, mr_o, \
                hits01_s, hits01_r, hits01_o, hits03_s, hits03_r, hits03_o, hits10_s, hits10_r, hits10_o)
        for (int i = 0; i < N; i++) {
            auto ranks = this->rank(model, sros[i]);

            double rank_s = get<0>(ranks);
            double rank_r = get<1>(ranks);
            double rank_o = get<2>(ranks);
            double rank_s_raw = get<3>(ranks);
            double rank_o_raw = get<4>(ranks);

            mrr_s += 1./rank_s;
            mrr_r += 1./rank_r;
            mrr_o += 1./rank_o;
            mrr_s_raw += 1./rank_s_raw;
            mrr_o_raw += 1./rank_o_raw;

            mr_s += rank_s;
            mr_r += rank_r;
            mr_o += rank_o;
            mr_s_raw += rank_s_raw;
            mr_o_raw += rank_o_raw;

            hits01_s += rank_s_raw <= 01;
            hits01_r += rank_r <= 01;
            hits01_o += rank_o_raw <= 01;

            hits03_s += rank_s_raw <= 03;
            hits03_r += rank_r <= 03;
            hits03_o += rank_o_raw <= 03;

            hits10_s += rank_s_raw <= 10;
            hits10_r += rank_r <= 10;
            hits10_o += rank_o_raw <= 10;
        }

        unordered_map<string, double> info;

        info["mrr_s"] = mrr_s / N;
        info["mrr_r"] = mrr_r / N;
        info["mrr_o"] = mrr_o / N;
        info["mrr_s_raw"] = mrr_s_raw / N;
        info["mrr_o_raw"] = mrr_o_raw / N;

        info["mr_s"] = mr_s / N;
        info["mr_r"] = mr_r / N;
        info["mr_o"] = mr_o / N;
        info["mr_s_raw"] = mr_s_raw / N;
        info["mr_o_raw"] = mr_o_raw / N;

        info["hits01_s"] = hits01_s / N; 
        info["hits01_r"] = hits01_r / N;
        info["hits01_o"] = hits01_o / N;

        info["hits03_s"] = hits03_s / N;
        info["hits03_r"] = hits03_r / N;
        info["hits03_o"] = hits03_o / N;
                                      
        info["hits10_s"] = hits10_s / N;
        info["hits10_r"] = hits10_r / N;
        info["hits10_o"] = hits10_o / N;

        return info;
    }

private:

    tuple<double, double, double, double, double> rank(const Model *model, const triplet& sro) {
        int rank_s = 1;
        int rank_r = 1;
        int rank_o = 1;

        int s = get<0>(sro);
        int r = get<1>(sro);
        int o = get<2>(sro);

        // XXX:
        // There might be degenerated cases when all output scores == 0, leading to perfect but meaningless results.
        // A quick fix is to add a small offset to the base_score.
        double base_score = model->score(s, r, o) - 1e-32;

        for (int ss = 0; ss < ne; ss++)
            if (model->score(ss, r, o) > base_score) rank_s++;

        for (int rr = 0; rr < nr; rr++)
            if (model->score(s, rr, o) > base_score) rank_r++;

        for (int oo = 0; oo < ne; oo++)
            if (model->score(s, r, oo) > base_score) rank_o++;

        int rank_s_raw = rank_s;
        int rank_o_raw = rank_o;

        for (auto ss : sro_bucket.or2s(o, r))
            if (model->score(ss, r, o) > base_score) rank_s--;

        for (auto oo : sro_bucket.sr2o(s, r))
            if (model->score(s, r, oo) > base_score) rank_o--;

        return make_tuple(rank_s, rank_r, rank_o, rank_s_raw, rank_o_raw);
    }
};

void pretty_print(const char* prefix, const unordered_map<string, double>& info) {
    printf("%s  MRR    \t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("mrr_s"),    100*info.at("mrr_r"),    100*info.at("mrr_o"));
    printf("%s  MRR_RAW\t%.2f\t%.2f\n", prefix, 100*info.at("mrr_s_raw"),    100*info.at("mrr_o_raw"));
    printf("%s  MR     \t%.2f\t%.2f\t%.2f\n", prefix, info.at("mr_s"), info.at("mr_r"), info.at("mr_o"));
    printf("%s  MR_RAW \t%.2f\t%.2f\n", prefix, info.at("mr_s_raw"), info.at("mr_o_raw"));
    printf("%s  Hits@01\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits01_s"), 100*info.at("hits01_r"), 100*info.at("hits01_o"));
    printf("%s  Hits@03\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits03_s"), 100*info.at("hits03_r"), 100*info.at("hits03_o"));
    printf("%s  Hits@10\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits10_s"), 100*info.at("hits10_r"), 100*info.at("hits10_o"));
}

void pretty_print2(const char* prefix, const unordered_map<string, double>& info, std::ofstream& fp) {
    fp << prefix << "  MRR    " << 100*info.at("mrr_s") <<  "  " << 100*info.at("mrr_r") <<  "  " << 100*info.at("mrr_o") << "\n";
    fp << prefix << "  MRR_RAW " << 100*info.at("mrr_s_raw") << "  " << 100*info.at("mrr_o_raw") << "\n";
    fp << prefix << "  MR     " << info.at("mr_s") << "  " << info.at("mr_r") << "  " << info.at("mr_o") << "\n";
    fp << prefix << "  MR_RAW " << info.at("mr_s_raw") << "  " << info.at("mr_o_raw") << "\n";
    fp << prefix << "  Hits@01 " << 100*info.at("hits01_s") << "  " << 100*info.at("hits01_r") << "  " << 100*info.at("hits01_o") << "\n";
    fp << prefix << "  Hits@03 " << 100*info.at("hits03_s") << "  " << 100*info.at("hits03_r") << "  " << 100*info.at("hits03_o") << "\n";
    fp << prefix << "  Hits@10 " << 100*info.at("hits10_s") << "  " << 100*info.at("hits10_r") << "  " << 100*info.at("hits10_o") << "\n";
}

// based on Google's word2vec
int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}


class DistMult : public Model {
    int nh;

public:
    DistMult(int ne, int nr, int nh, double eta, double gamma) : Model(eta, gamma) {
        this->nh = nh;

        E = uniform_matrix(ne, nh, -init_b, init_b);
        R = uniform_matrix(nr, nh, -init_b, init_b);
        E_g = const_matrix(ne, nh, init_e);
        R_g = const_matrix(nr, nh, init_e);
    }

    double score(int s, int r, int o) const {
        double dot = 0;
        for (int i = 0; i < nh; i++)
            dot += E[s][i] * R[r][i] * E[o][i];
        return dot;
    }

    void score_grad(
            int s,
            int r,
            int o,
            vector<double>& d_s, 
            vector<double>& d_r, 
            vector<double>& d_o) {

        for (int i = 0; i < nh; i++) {
            d_s[i] = R[r][i] * E[o][i];
            d_r[i] = E[s][i] * E[o][i]; 
            d_o[i] = E[s][i] * R[r][i];
        }
    }
};


class Complex : public Model {
    int nh;

public:
    Complex(int ne, int nr, int nh, double eta, double gamma) : Model(eta, gamma) {
        assert( nh % 2 == 0 );
        this->nh = nh;

        E = uniform_matrix(ne, nh, -init_b, init_b);
        R = uniform_matrix(nr, nh, -init_b, init_b);
        E_g = const_matrix(ne, nh, init_e);
        R_g = const_matrix(nr, nh, init_e);
    }

    double score(int s, int r, int o) const {
        double dot = 0;

        int nh_2 = nh/2;
        for (int i = 0; i < nh_2; i++) {
            dot += R[r][i]      * E[s][i]      * E[o][i];
            dot += R[r][i]      * E[s][nh_2+i] * E[o][nh_2+i];
            dot += R[r][nh_2+i] * E[s][i]      * E[o][nh_2+i];
            dot -= R[r][nh_2+i] * E[s][nh_2+i] * E[o][i];
        }
        return dot;
    }

    void score_grad(
        int s,
        int r,
        int o,
        vector<double>& d_s, 
        vector<double>& d_r, 
        vector<double>& d_o) {

        int nh_2 = nh/2;
        for (int i = 0; i < nh_2; i++) {
            // re
            d_s[i] = R[r][i] * E[o][i] + R[r][nh_2+i] * E[o][nh_2+i];
            d_r[i] = E[s][i] * E[o][i] + E[s][nh_2+i] * E[o][nh_2+i];
            d_o[i] = R[r][i] * E[s][i] - R[r][nh_2+i] * E[s][nh_2+i];
            // im
            d_s[nh_2+i] = R[r][i] * E[o][nh_2+i] - R[r][nh_2+i] * E[o][i];
            d_r[nh_2+i] = E[s][i] * E[o][nh_2+i] - E[s][nh_2+i] * E[o][i];
            d_o[nh_2+i] = R[r][i] * E[s][nh_2+i] + R[r][nh_2+i] * E[s][i];
        }
    }
};


class Analogy : public Model {
    int nh1;
    int nh2;

public:
    Analogy(int ne, int nr, int nh, int num_scalar, double eta, double gamma) : Model(eta, gamma) {
        this->nh1 = num_scalar;
        this->nh2 = nh - num_scalar;
        assert( this->nh2 % 2 == 0 );
        // creates a matrix of size number entities by embedding dim 
        // initialized to random numbers between -0.01 and 0.01
        E = uniform_matrix(ne, nh, -init_b, init_b);
        // creates a matrix of size number relations by embedding dim 
        // initialized to random numbers between -0.01 and 0.01
        R = uniform_matrix(nr, nh, -init_b, init_b);
        // creates a matrix of size number entities by embedding dim 
        // initialized to constant 0.000006 (used in adagrad learning rate)
        E_g = const_matrix(ne, nh, init_e);
        // creates a matrix of size number relations by embedding dim 
        // initialized to constant 0.000006 (used in adagrad learning rate)
        R_g = const_matrix(nr, nh, init_e);
    }

    double score(int s, int r, int o) const {
        // outputs the score for a particular s,r,o triplet
        double dot = 0;

        int i = 0;
        for (; i < nh1; i++)
            dot += E[s][i] * R[r][i] * E[o][i];

        int nh2_2 = nh2/2;
        for (; i < nh1 + nh2_2; i++) {
            dot += R[r][i]       * E[s][i]       * E[o][i];
            dot += R[r][i]       * E[s][nh2_2+i] * E[o][nh2_2+i];
            dot += R[r][nh2_2+i] * E[s][i]       * E[o][nh2_2+i];
            dot -= R[r][nh2_2+i] * E[s][nh2_2+i] * E[o][i];
        }

        return dot;
    }

    void score_grad(
        // outputs the gradient of the loss for a particular s,r,o triplet
            int s,
            int r,
            int o,
            vector<double>& d_s, 
            vector<double>& d_r, 
            vector<double>& d_o) {

        int i = 0;
        for (; i < nh1; i++) {
            d_s[i] = R[r][i] * E[o][i];
            d_r[i] = E[s][i] * E[o][i]; 
            d_o[i] = E[s][i] * R[r][i];
        }

        int nh2_2 = nh2/2;
        for (; i < nh1 + nh2_2; i++) {
            // re
            d_s[i] = R[r][i] * E[o][i] + R[r][nh2_2+i] * E[o][nh2_2+i];
            d_r[i] = E[s][i] * E[o][i] + E[s][nh2_2+i] * E[o][nh2_2+i];
            d_o[i] = R[r][i] * E[s][i] - R[r][nh2_2+i] * E[s][nh2_2+i];
            // im
            d_s[nh2_2+i] = R[r][i] * E[o][nh2_2+i] - R[r][nh2_2+i] * E[o][i];
            d_r[nh2_2+i] = E[s][i] * E[o][nh2_2+i] - E[s][nh2_2+i] * E[o][i];
            d_o[nh2_2+i] = R[r][i] * E[s][nh2_2+i] + R[r][nh2_2+i] * E[s][i];
        }

    }
};


int main(int argc, char **argv) {
    // default options
    string  dataset     =  "thor_eg_all_0"; // training set
    string  cfile       =  "thor"; // training set
    string  algorithm   =  "Analogy"; // model type
    int     embed_dim   =  100; // dimensionality of the embedding
    double  eta         =  0.1; // related to learning rate
    double  gamma       =  1e-3; // related to gradient
    int     neg_ratio   =  9; // number of negative examples to see for every
    // positive example
    int     num_epoch   =  500; // number of epochs
    int     num_thread  =  32; // number of threads
    int     eval_freq   =  10; // how often to evaluate while training
    string  model_path; // path to output- (train) or input- (test) model
    bool    prediction  = false; // whether testing or training
    int     num_scalar  = embed_dim / 2.0;
    int     train_size  = 0;
    // parses all the ANALOGY arguments
    int i;
    if ((i = ArgPos((char *)"-algorithm",  argc, argv)) > 0)  algorithm   =  string(argv[i+1]);
    if ((i = ArgPos((char *)"-embed_dim",  argc, argv)) > 0)  embed_dim   =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-eta",        argc, argv)) > 0)  eta         =  atof(argv[i+1]);
    if ((i = ArgPos((char *)"-gamma",      argc, argv)) > 0)  gamma       =  atof(argv[i+1]);
    if ((i = ArgPos((char *)"-neg_ratio",  argc, argv)) > 0)  neg_ratio   =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-num_epoch",  argc, argv)) > 0)  num_epoch   =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-num_thread", argc, argv)) > 0)  num_thread  =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-eval_freq",  argc, argv)) > 0)  eval_freq   =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-model_path", argc, argv)) > 0)  model_path  =  string(argv[i+1]);
    if ((i = ArgPos((char *)"-dataset",    argc, argv)) > 0)  dataset     =  string(argv[i+1]);
    if ((i = ArgPos((char *)"-cfile",      argc, argv)) > 0)  cfile       =  string(argv[i+1]);
    if ((i = ArgPos((char *)"-prediction", argc, argv)) > 0)  prediction  =  true;
    if ((i = ArgPos((char *)"-num_scalar", argc, argv)) > 0)  num_scalar  =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-train_size", argc, argv)) > 0)  train_size  =  atoi(argv[i+1]);
    num_scalar  = embed_dim / 2.0;
    // lists all the ANALOGY arguments
    printf("dataset     =  %s\n", dataset.c_str());
    printf("cfile       =  %s\n", cfile.c_str());
    printf("algorithm   =  %s\n", algorithm.c_str());
    printf("embed_dim   =  %d\n", embed_dim);
    printf("eta         =  %e\n", eta);
    printf("gamma       =  %e\n", gamma);
    printf("neg_ratio   =  %d\n", neg_ratio);
    printf("num_epoch   =  %d\n", num_epoch);
    printf("num_thread  =  %d\n", num_thread);
    printf("eval_freq   =  %d\n", eval_freq);
    printf("model_path  =  %s\n", model_path.c_str());
    printf("num_scalar  =  %d\n", num_scalar);
    printf("train_size     =  %d\n", train_size);
    // gets entities as strings
    string cdir = "./constants/";
    string ddir = "./datasets/";
    vector<string> ents = read_first_column(cdir+cfile+"_entities.txt");
    // gets relations as strings
    vector<string> rels = read_first_column(cdir+cfile+"_relations.txt");
    // creates mapping between entities and unique ID
    unordered_map<string, int> ent_map = create_id_mapping(ents);
    // creates mapping between relations and unique ID
    unordered_map<string, int> rel_map = create_id_mapping(rels);
    // records number of entities and number of relations
    int ne = ent_map.size();
    int nr = rel_map.size();
    // loads the train, test, and validation triplets from each dataset
    vector<triplet> sros_tr = 
        create_sros(ddir+dataset+"_train.txt",ent_map,rel_map);
    vector<triplet> sros_va = 
        create_sros(ddir+dataset+"_valid.txt",ent_map,rel_map);
    vector<triplet> sros_te = 
        create_sros(ddir+dataset+"_test.txt",ent_map,rel_map);
    vector<triplet> sros_al;
    // store all the triplets in sros_al
    sros_al.insert(sros_al.end(), sros_tr.begin(), sros_tr.end());
    sros_al.insert(sros_al.end(), sros_va.begin(), sros_va.end());
    sros_al.insert(sros_al.end(), sros_te.begin(), sros_te.end());
    // creates a 'bucket' object of all s,r,o triplets, used later
    SROBucket sro_bucket_al(sros_al);
    // selects the model that will be used
    Model *model = NULL;
    if (algorithm == "DistMult") 
        model = new DistMult(ne,nr,embed_dim,eta,gamma);
    if (algorithm == "Complex") 
        model = new Complex(ne,nr,embed_dim,eta,gamma);
    // initializes the ANALOGY model for use
    if (algorithm == "Analogy") 
        model = new Analogy(ne,nr,embed_dim,num_scalar,eta,gamma);
    assert(model != NULL);
    
    // checks if we are in the testing phase of program
    if (prediction) {
        // if so creates the testing evaluator
        Evaluator evaluator_te(ne, nr, sros_te, sro_bucket_al);
        //Evaluator evaluator_te(ne, nr, sros_tr, sro_bucket_al);
        model->load(model_path);
        auto info_te = evaluator_te.evaluate(model, -1);
        //evaluator_te.fake_eval(model);
        pretty_print("TE", info_te);
        return 0;
    }
    
    // evaluator for validation data
    Evaluator evaluator_va(ne, nr, sros_va, sro_bucket_al);
    // evaluator for training data
    Evaluator evaluator_tr(ne, nr, sros_tr, sro_bucket_al);


    // thread-specific negative samplers
    vector<NegativeSampler> neg_samplers;
    for (int tid = 0; tid < num_thread; tid++) {
        // creates a bunch of random entity/relation generators
        neg_samplers.push_back( NegativeSampler(ne, nr, rand() ^ tid) );
    }

    int N = sros_tr.size(); // N is number of examples in training set
    vector<int> pi = range(N); // pi is the range of numbers in N

    clock_t start;
    double elapse_tr = 0;
    double elapse_ev = 0;
    double best_mrr = 0;
    
    double last_mrr = 0;
    
    double last_hits = 0;
    double last_hito = 0;
    double last_hitr = 0;
    //int model_count = 0;
    
    omp_set_num_threads(num_thread); // tells omp lib num of threads to use

    start = omp_get_wtime();
    for (int epoch = 0; epoch < num_epoch; epoch++) {
        // goes through the full number of epochs for training
        if (epoch % eval_freq == 0) {
            // evaluation
            //start = omp_get_wtime();
            auto info_tr = evaluator_tr.evaluate(model, 2048);
            auto info_va = evaluator_va.evaluate(model, 2048);
            //elapse_ev = omp_get_wtime() - start;

            // save the best model to disk
            double curr_mrr_raw = (info_va["mrr_s_raw"]+info_va["mrr_o_raw"])/2;
            double curr_hits = (info_va["hits10_o"] + info_va["hits10_s"])/2;
            double curr_mrr = (info_va["mrr_s"] + info_va["mrr_o"])/2;
            
            if (curr_mrr_raw > best_mrr) {
                best_mrr = curr_mrr_raw;
                if ( !model_path.empty() )
                    model->save(model_path+".model");
            }
            
            
            printf("\n");
            printf("            EV Elapse    %f\n", elapse_ev);
            printf("======================================\n");
            pretty_print("TR", info_tr);
            printf("\n");
            pretty_print("VA", info_va);
            printf("\n");
            printf("VA  MRR_BEST_RAW    %.2f\n", 100*best_mrr);
            printf("VA  HITS_CURR    %.2f\n", 100*curr_hits);
            printf("VA  MRR_CURR_RAW    %.2f\n", 100*curr_mrr_raw);
            printf("VA  MRR_CURR    %.2f\n", 100*curr_mrr);
            printf("\n");
            
            // write to log file
            ofstream logfile;
            logfile.open("logfile.txt",std::ios_base::app);
            logfile << "\n";
            logfile << "     " << dataset << "   EV Elapse    %f\n"<< elapse_ev;
            logfile << "======================================\n";
            pretty_print2("TR", info_tr, logfile);
            logfile << "\n";
            pretty_print2("VA", info_va, logfile);
            logfile << "\n";
            
            logfile << "VA  MRR_BEST_RAW    " << 100*best_mrr << endl;
            logfile << "VA  HITS_CURR    " << 100*curr_hits << endl;
            logfile << "VA  MRR_CURR_RAW    " << 100*curr_mrr_raw << endl;
            logfile << "VA  MRR_CURR    " << 100*curr_mrr << endl;
            
            logfile << "\n";
            
            cout << "mrr diff " 
                << abs(100.0*curr_mrr_raw - 100.0*last_mrr) << endl;
            cout << "hit s diff " 
                << abs(100.0*info_va.at("hits01_s")-100.0*last_hits) << endl;
            cout << "hit r diff " 
                << abs(100.0*info_va.at("hits01_r")-100.0*last_hitr) << endl;
            cout << "hit o diff " 
                << abs(100.0*info_va.at("hits01_o")-100.0*last_hito) << endl;
            
            
            /*if ( (abs(100.0*curr_mrr_raw - 100.0*last_mrr) < 2.0) && 
                    (abs(100.0*info_va.at("hits01_s")-100.0*last_hits)<2.0) &&
                    (abs(100.0*info_va.at("hits01_r")-100.0*last_hitr)<2.0) &&
                    (abs(100.0*info_va.at("hits01_o")-100.0*last_hito)<2.0) ) {
                return 0;
            }*/
            last_hits = info_va.at("hits01_s");
            last_hito = info_va.at("hits01_o");
            last_hitr = info_va.at("hits01_r");
            last_mrr = curr_mrr_raw;
        }
        
        /*if (epoch < 10) {
            std::stringstream model_path_temp;
            std::stringstream model_number;
            model_number << std::setw(2) << std::setfill('0') << model_count;
            model_path_temp << model_path << model_number.str();
            model_count += 1;
            if ( !model_path_temp.str().empty() )
                model->save(model_path_temp.str() + ".model");
        } else if (epoch < 100) {
            if (epoch % 10 == 0){
                std::stringstream model_path_temp;
                std::stringstream model_number;
                model_number<<std::setw(2)<<std::setfill('0') << model_count;
                model_path_temp << model_path << model_number.str();
                model_count += 1;
                if ( !model_path_temp.str().empty() )
                    model->save(model_path_temp.str() + ".model");
            }
        }*/
        
        // shuffles all the numbers corresponding to each example
        shuffle(pi.begin(), pi.end(), GLOBAL_GENERATOR);

        //start = omp_get_wtime();
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            // goes through each example
            triplet sro = sros_tr[pi[i]];
            // extracts the subj, rel, obj
            int s = get<0>(sro);
            int r = get<1>(sro);
            int o = get<2>(sro);

            int tid = omp_get_thread_num();

            // trains on 1 positive example
            model->train(s, r, o, true);

            // trains on neg_ratio*3 negative examples
            for (int j = 0; j < neg_ratio; j++) {
                // creates 'negative' ss, rr, oo for sample (XXX not garunteed)
                int oo = neg_samplers[tid].random_entity();
                int ss = neg_samplers[tid].random_entity();
                int rr = neg_samplers[tid].random_relation();

                // XXX: it is empirically beneficial to carry out updates even
                // if oo == o || ss == s.
                // This might be related to regularization.
                model->train(s, r, oo, false);
                model->train(ss, r, o, false);
                model->train(s, rr, o, false);   // this improves MR slightly
            }
        }
        //elapse_tr = omp_get_wtime() - start;
        //printf("Epoch %03d   TR Elapse    %f\n", epoch, elapse_tr);
    }
    elapse_tr = omp_get_wtime() - start;
    printf("Elapse    %f\n", elapse_tr);

    return 0;
}

