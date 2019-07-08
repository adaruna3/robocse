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
#include <stdio.h>
#include <set>


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

    getline(ifs, line); // skip first line
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
    getline(ifs, line); // skip first line
    while (getline(ifs, line)) { // go through all lines in dataset
        stringstream ss(line);
        getline(ss,s,',');
        getline(ss,r,',');
        getline(ss,o,',');
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
    unordered_map<int64_t, int> __counts; // counts of all s,r,o hashes
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
            // adds s,r,o hash to s,r,o hash list and counts memory
            int64_t __sro = hash(s, r, o);
            __sros.insert(__sro);
            if (__counts.find(__sro) == __counts.end()) {
                __counts[__sro] = 1;
            } else {
                __counts[__sro] += 1;
            }
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

    bool score(int a, int b, int c) const {
        // gets numbers of s,r,o
        int count = 0;
        if (__counts.find(hash(a,b,c)) != __counts.end()) {
            count = __counts.at(hash(a,b,c));
        }
        return count;
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
    vector<vector<int>> negs;
    int num_ent;

public:
    NegativeSampler(int ne, int nr, int seed,
                    const unordered_map<string,int>& ent_map,
                    const unordered_map<string, int>& rel_map,
                    const vector<triplet>& sros) {
        // creates a random num generator in range of relations
        unif_r = uniform_int_distribution<int>(0, nr-1);
        generator = default_random_engine(seed);
        num_ent = ne;

        // resizes the truem matrix
        vector<vector<vector<int>>> truem; // CWA negative sampling!
        truem.resize(nr);
        for (int i=0;i<nr;i++) { // loops through relations
            truem[i].resize(ne);
            for (int j=0;j<ne;j++) { // loops through subjects
                truem[i][j].resize(ne);
            }
        }
        // uploads the truem matrix with training triples
        for (int i=0;i<nr;i++) { // loops through relations
            for (int j=0;j<ne;j++) { // loops through subjects
                for (int k=0;k<ne;k++) { // loops through objects
                    triplet current = make_tuple(j,i,k);
                    if (find(sros.begin(),sros.end(),current) != sros.end()) {
                        truem[i][j][k] = 1;
                    }
                }
            }
        }
        // generates a lists of random SUBJECT true negatives by type (CWA)
        negs.resize(nr);
        for (int rel=0;rel<nr;rel++) {
            for (int subj=0;subj<ne;subj++) {
                int flag = 0;
                for (int obj=0;obj<ne;obj++) {
                    if (truem[rel][subj][obj] == 1) {
                        flag = 1;
                        break;
                    }
                }
                if (flag == 0) {
                    negs[rel].push_back(subj);
                }
            }
        }
    }

    int random_s_entity(int rel) {
        // returns a random entity number
        /*if ( (rel == 0) or (rel == 1) ) {
            unif_e = uniform_int_distribution<int>(0, negs[rel].size()-1);
            return negs[rel][unif_e(generator)];
        } else {
            unif_e = uniform_int_distribution<int>(0, num_ent-1);
            return unif_e(generator);
        }*/
        unif_e = uniform_int_distribution<int>(0, negs[rel].size()-1);
        return negs[rel][unif_e(generator)];
    }

    int random_o_entity(int rel) {
        // returns a random entity number
        unif_e = uniform_int_distribution<int>(0, num_ent-1);
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

public:
    vector<vector<double>> E; // entity embeddings
    vector<vector<double>> R; // relation embeddings
    vector<vector<double>> E_g; // entity matrix for adagrad
    vector<vector<double>> R_g; // relation matrix for adagrad

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
        assert(!ifs.fail());

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

class RandomModel {
    uniform_int_distribution<int> unif_e; // random entity generator
    uniform_int_distribution<int> unif_r; // random relation generator
    default_random_engine generator; // random seed for generators

public:
    RandomModel(int ne, int nr, int seed) :
        unif_e(0, ne-1), unif_r(0, nr-1), generator(seed) {}

    void score(int s, int r, int o, int& rs, int& rr, int& ro, string type) {
        // randomly rank and s,r,o
        // TODO update to include type-specific random (logic basically)
        if (type == "random"){
            rs = unif_e(generator);
            ro = unif_e(generator);
            rr = unif_r(generator);
        }
        return;
    }

};

class Evaluator {
    int ne;
    int nr;
    const vector<triplet>& sros;
    typedef tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>> ranked_list_set;

public:
    Evaluator(int ne, int nr, const vector<triplet>& sros) :
        // sets the number of relations and entities, also gives the set of 
        // s,r,o used to either test, train, or validate and the complete s,r,o
        // bucket        
        ne(ne), nr(nr), sros(sros) {}

    std::vector<std::vector<std::vector<double>>> evaluate(const Model *model, const SROBucket *bucket, const SROBucket* skip_bucket, int truncate) {
        // complete training set size
        int N = this->sros.size();
        if (truncate > 0)
            N = min(N, truncate);

        // declares counters for metrics
        vector<vector<double>> totals;
        vector<vector<vector<double>>> metric;
        totals.resize(3);
        metric.resize(3);
        for(int i = 0; i < 3; i++){
            totals[i].resize(nr);
            metric[i].resize(nr);
            for(int j = 0; j < nr; j++){
                totals[i][j] = 0.0;
                metric[i][j].resize(5);
                for(int k = 0; k < 5; k++){
                    metric[i][j][k] = 0.0;
                }
            }
        }

        // calculates metrics in parallel
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            auto ranks1 = this->rank(model, bucket, skip_bucket, sros[i]);
            auto ranks2 = this->rank(bucket, skip_bucket, sros[i]);
            double rank_s = abs(get<0>(ranks1)-get<0>(ranks2))+1.0;
            double rank_r = abs(get<1>(ranks1)-get<1>(ranks2))+1.0;
            double rank_o = abs(get<2>(ranks1)-get<2>(ranks2))+1.0;
            auto ranked_lists1 = this->get_ranked_list(bucket,skip_bucket,sros[i]);
            int list_size_s = get<0>(ranked_lists1).size();
            int list_size_r = get<1>(ranked_lists1).size();
            int list_size_o = get<2>(ranked_lists1).size();
            tuple<int,int,int> list_sizes = make_tuple(list_size_s, list_size_r, list_size_o);
            auto ranked_lists2 = this->get_ranked_list(model,skip_bucket,sros[i],list_sizes);
            auto average_precisions = this->average_precision(ranked_lists1,ranked_lists2);
            double average_precision_s = get<0>(average_precisions);
            double average_precision_r = get<1>(average_precisions);
            double average_precision_o = get<2>(average_precisions);
            int r = get<1>(sros[i]);
            #pragma omp critical
            {
                // MRR*
                metric[0][r][0] += 1.0/rank_s;
                metric[1][0][0] += 1.0/rank_r;
                metric[2][r][0] += 1.0/rank_o;
                // HITS10*
                metric[0][r][1] += rank_s <= 10;
                metric[1][0][1] += rank_r <= 10;
                metric[2][r][1] += rank_o <= 10;
                // HITS5*
                metric[0][r][2] += rank_s <= 05;
                metric[1][0][2] += rank_r <= 05;
                metric[2][r][2] += rank_o <= 05;
                // HITS1*
                metric[0][r][3] += rank_s <= 01;
                metric[1][0][3] += rank_r <= 01;
                metric[2][r][3] += rank_o <= 01;
                // AVERAGE PRECISION*
                metric[0][r][4] += average_precision_s;
                metric[1][0][4] += average_precision_r;
                metric[2][r][4] += average_precision_o;
                // Totals
                totals[0][r] += 1.0;
                totals[1][0] += 1.0;
                totals[2][r] += 1.0;
            }
        }

        for(int i = 0; i < 3; i++) {
            if (i == 1) {
                metric[i][0][0] = metric[i][0][0] / totals[i][0];
                metric[i][0][1] = metric[i][0][1] / totals[i][0];
                metric[i][0][2] = metric[i][0][2] / totals[i][0];
                metric[i][0][3] = metric[i][0][3] / totals[i][0];
                metric[i][0][4] = metric[i][0][4] / totals[i][0];
                continue;
            }
            for(int j = 0; j < nr; j++) {
                metric[i][j][0] = metric[i][j][0] / totals[i][j];
                metric[i][j][1] = metric[i][j][1] / totals[i][j];
                metric[i][j][2] = metric[i][j][2] / totals[i][j];
                metric[i][j][3] = metric[i][j][3] / totals[i][j];
                metric[i][j][4] = metric[i][j][4] / totals[i][j];
            }
        }

        return metric;
    }

    /*std::vector<std::vector<std::vector<double>>> evaluate(RandomModel *model, const SROBucket *bucket, int truncate) {
        // evaluates using the random model
        // TODO update this section of code to include MAP in evaluation!!!
        // complete training set size
        int N = this->sros.size();
        if (truncate > 0)
            N = min(N, truncate);

        // declares counters for metrics
        vector<vector<double>> totals;
        vector<vector<vector<double>>> metric;
        totals.resize(3);
        metric.resize(3);
        for(int i = 0; i < 3; i++){
            totals[i].resize(nr);
            metric[i].resize(nr);
            for(int j = 0; j < nr; j++){
                totals[i][j] = 0.0;
                metric[i][j].resize(4);
                for(int k = 0; k < 4; k++){
                    metric[i][j][k] = 0.0;
                }
            }
        }

        // calculates metrics in parallel
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            auto ranks1 = this->rank(model, sros[i]);
            auto ranks2 = this->rank(bucket, sros[i]);
            double rank_s = abs(get<0>(ranks1)-get<0>(ranks2))+1.0;
            double rank_r = abs(get<1>(ranks1)-get<1>(ranks2))+1.0;
            double rank_o = abs(get<2>(ranks1)-get<2>(ranks2))+1.0;
            int r = get<1>(sros[i]);
            #pragma omp critical
            {
                // MRR*
                metric[0][r][0] += 1.0/rank_s;
                metric[1][0][0] += 1.0/rank_r;
                metric[2][r][0] += 1.0/rank_o;
                // HITS10*
                metric[0][r][1] += rank_s <= 10;
                metric[1][0][1] += rank_r <= 10;
                metric[2][r][1] += rank_o <= 10;
                // HITS5*
                metric[0][r][2] += rank_s <= 05;
                metric[1][0][2] += rank_r <= 05;
                metric[2][r][2] += rank_o <= 05;
                // HITS1*
                metric[0][r][3] += rank_s <= 01;
                metric[1][0][3] += rank_r <= 01;
                metric[2][r][3] += rank_o <= 01;
                // Totals
                totals[0][r] += 1.0;
                totals[1][0] += 1.0;
                totals[2][r] += 1.0;
            }
        }

        for(int i = 0; i < 3; i++) {
            if (i == 1) {
                metric[i][0][0] = metric[i][0][0] / totals[i][0];
                metric[i][0][1] = metric[i][0][1] / totals[i][0];
                metric[i][0][2] = metric[i][0][2] / totals[i][0];
                metric[i][0][3] = metric[i][0][3] / totals[i][0];
                continue;
            }
            for(int j = 0; j < nr; j++) {
                metric[i][j][0] = metric[i][j][0] / totals[i][j];
                metric[i][j][1] = metric[i][j][1] / totals[i][j];
                metric[i][j][2] = metric[i][j][2] / totals[i][j];
                metric[i][j][3] = metric[i][j][3] / totals[i][j];
            }
        }

        return metric;
    }*/

private:

    tuple<double, double, double> rank(const SROBucket* model, const SROBucket* skip_bucket, const triplet& sro) {
        // ranks with ground truth
        int rank_s = 1;
        int rank_r = 1;
        int rank_o = 1;

        int s = get<0>(sro);
        int r = get<1>(sro);
        int o = get<2>(sro);

        double base_score = double(model->score(s, r, o));

        for (int ss = 0; ss < ne; ss++) {
            if (skip_bucket->score(ss,r,o) != 0) continue;
            if (model->score(ss, r, o) > base_score) rank_s++;
        }

        for (int rr = 0; rr < nr; rr++) {
            if (skip_bucket->score(s,rr,o) != 0) continue;
            if (model->score(s, rr, o) > base_score) rank_r++;
        }


        for (int oo = 0; oo < ne; oo++) {
            if (skip_bucket->score(s,r,oo) != 0) continue;
            if (model->score(s, r, oo) > base_score) rank_o++;
        }

        return make_tuple(rank_s, rank_r, rank_o);
    }

    tuple<double, double, double> rank(const Model* model, const SROBucket* bucket, const SROBucket* skip_bucket, const triplet& sro) {
        // ranks with robocse
        int rank_s = 1;
        int rank_r = 1;
        int rank_o = 1;

        int s = get<0>(sro);
        int r = get<1>(sro);
        int o = get<2>(sro);

        // XXX:
        // There might be degenerated cases when all output scores == 0, leading to perfect but meaningless results.
        // A quick fix is to add a small offset to the base_score.
        double base_score = double(model->score(s, r, o)) - 1e-32;
        double ground_truth_base_score = double(bucket->score(s, r, o));

        for (int ss = 0; ss < ne; ss++) {
            if (skip_bucket->score(ss,r,o) != 0) continue;
            if (bucket->score(ss,r,o) != ground_truth_base_score) {
                if (model->score(ss,r,o) > base_score) rank_s++;
            }
        }
        for (int rr = 0; rr < nr; rr++) {
            if (skip_bucket->score(s,rr,o) != 0) continue;
            if (bucket->score(s,rr,o) != ground_truth_base_score) {
                if (model->score(s,rr,o) > base_score) rank_r++;
            }
        }
        for (int oo = 0; oo < ne; oo++) {
            if (skip_bucket->score(s,r,oo) != 0) continue;
            if (bucket->score(s,r,oo) != ground_truth_base_score) {
                if (model->score(s,r,oo) > base_score) rank_o++;
            }
        }

        return make_tuple(rank_s, rank_r, rank_o);
    }

    tuple<double, double, double> rank(RandomModel *model, const triplet& sro) {
        // ranks completely randomly
        int rank_s;
        int rank_r;
        int rank_o;
        int s = get<0>(sro);
        int r = get<1>(sro);
        int o = get<2>(sro);

        model->score(s, r, o, rank_s, rank_r, rank_o, "random");

        return make_tuple(rank_s, rank_r, rank_o);
    }

    ranked_list_set get_ranked_list(const SROBucket *model, const SROBucket* skip_bucket, const triplet& sro) {
        // gets the ranked list for ground truth
        vector<vector<double>> ranked_list_s;
        vector<vector<double>> ranked_list_r;
        vector<vector<double>> ranked_list_o;

        int s = get<0>(sro);
        int r = get<1>(sro);
        int o = get<2>(sro);

        // gets ranked lists for each query type
        for (int ss = 0; ss < ne; ss++) {
            if (skip_bucket->score(ss,r,o) != 0) continue;
            double score;
            score = double(model->score(ss, r, o));
            if (score > 0) {
                vector<double> rank{double(ss),score};
                ranked_list_s.push_back(rank);
            }
        }
        for (int rr = 0; rr < nr; rr++) {
            if (skip_bucket->score(s,rr,o) != 0) continue;
            double score;
            score = double(model->score(s, rr, o));
            if (score > 0) {
                vector<double> rank{double(rr),score};
                ranked_list_r.push_back(rank);
            }
        }
        for (int oo = 0; oo < ne; oo++) {
            if (skip_bucket->score(s,r,oo) != 0) continue;
            double score;
            score = double(model->score(s, r, oo));
            if (score > 0) {
                vector<double> rank{double(oo),score};
                ranked_list_o.push_back(rank);
            }
        }
        // sort the ranked lists
        sort(ranked_list_s.begin(),ranked_list_s.end(),[](vector<double> a, vector<double> b) {
            return (a.at(1) > b.at(1));
        });
        sort(ranked_list_r.begin(),ranked_list_r.end(),[](vector<double> a, vector<double> b) {
            return (a.at(1) > b.at(1));
        });
        sort(ranked_list_o.begin(),ranked_list_o.end(),[](vector<double> a, vector<double> b) {
            return (a.at(1) > b.at(1));
        });
        return make_tuple(ranked_list_s,ranked_list_r,ranked_list_o);
    }


    ranked_list_set get_ranked_list(const Model* model, const SROBucket* skip_bucket, const triplet& sro, const tuple<int,int,int>& list_sizes) {
        // gets ranked list for robocse
        vector<vector<double>> ranked_list_s;
        vector<vector<double>> ranked_list_r;
        vector<vector<double>> ranked_list_o;

        int s = get<0>(sro);
        int r = get<1>(sro);
        int o = get<2>(sro);
        int list_size_s = get<0>(list_sizes);
        int list_size_r = get<1>(list_sizes);
        int list_size_o = get<2>(list_sizes);

        // gets ranked lists for each query type
        for (int ss = 0; ss < ne; ss++) {
            if (skip_bucket->score(ss,r,o) != 0) continue;
            double score;
            score = model->score(ss, r, o);
            vector<double> rank{double(ss),score};
            ranked_list_s.push_back(rank);
        }
        for (int rr = 0; rr < nr; rr++) {
            if (skip_bucket->score(s,rr,o) != 0) continue;
            double score;
            score = model->score(s, rr, o);
            vector<double> rank{double(rr),score};
            ranked_list_r.push_back(rank);
        }
        for (int oo = 0; oo < ne; oo++) {
            if (skip_bucket->score(s,r,oo) != 0) continue;
            double score;
            score = model->score(s, r, oo);
            vector<double> rank{double(oo),score};
            ranked_list_o.push_back(rank);
        }
        // sort the ranked lists
        sort(ranked_list_s.begin(),ranked_list_s.end(),[](vector<double> a, vector<double> b) {
            return (a.at(1) > b.at(1));
        });
        sort(ranked_list_r.begin(),ranked_list_r.end(),[](vector<double> a, vector<double> b) {
            return (a.at(1) > b.at(1));
        });
        sort(ranked_list_o.begin(),ranked_list_o.end(),[](vector<double> a, vector<double> b) {
            return (a.at(1) > b.at(1));
        });
        vector<vector<double>> final_s(ranked_list_s.begin(),ranked_list_s.begin()+list_size_s);
        vector<vector<double>> final_r(ranked_list_r.begin(),ranked_list_r.begin()+list_size_r);
        vector<vector<double>> final_o(ranked_list_o.begin(),ranked_list_o.begin()+list_size_o);
        return make_tuple(final_s,final_r,final_o);
    }

    tuple<double, double, double> average_precision(ranked_list_set ground_truth_set,ranked_list_set result_set) {
        double avg_precision_s;
        double avg_precision_r;
        double avg_precision_o;
        vector<vector<double>> ground_truth_s;
        vector<vector<double>> ground_truth_r;
        vector<vector<double>> ground_truth_o;
        vector<vector<double>> result_set_s;
        vector<vector<double>> result_set_r;
        vector<vector<double>> result_set_o;
        ground_truth_s = get<0>(ground_truth_set);
        ground_truth_r = get<1>(ground_truth_set);
        ground_truth_o = get<2>(ground_truth_set);
        result_set_s = get<0>(result_set);
        result_set_r = get<1>(result_set);
        result_set_o = get<2>(result_set);
        avg_precision_s = calculate_average_precision(ground_truth_s,result_set_s);
        avg_precision_r = calculate_average_precision(ground_truth_r,result_set_r);
        avg_precision_o = calculate_average_precision(ground_truth_o,result_set_o);
        return make_tuple(avg_precision_s,avg_precision_r,avg_precision_o);
    }

    double calculate_average_precision(vector<vector<double>> ground_truth, vector<vector<double>> results) {
        double avg_precision = 0.0;
        double num_correct = 0.0;
        double num_total = 0.0;
        unsigned ground_truth_index = 0;
        unsigned result_set_index = 0;
        while (ground_truth_index<ground_truth.size()) {
            set<double> bin;
            double current_rank = ground_truth[ground_truth_index][1];
            while (ground_truth_index<ground_truth.size()) {
                if (ground_truth[ground_truth_index][1] == current_rank) {
                    bin.insert(ground_truth[ground_truth_index][0]);
                    ground_truth_index++;
                    continue;
                } else {
                    break;
                }
            }
            while (result_set_index<bin.size()) {
                if (bin.count(results[result_set_index][0]) > 0) {
                    num_correct += 1.0;
                }
                num_total += 1.0;
                avg_precision += num_correct / num_total;
                result_set_index++;
            }
        }
        return avg_precision / double(results.size());
    }
};

void eval_log(const char* prefix, const vector<vector<vector<double>>>& info) {
    FILE * pFile;
    pFile = fopen("logfile.txt","a");
    fprintf(pFile,"------------------%s---------------------\n", prefix);
    fprintf(pFile,"| Query | Relation |   MRR* | Hits10*| Hits5* | Hits1* |   MAP* |\n");
    fprintf(pFile,"|  Sro  | atLoc    | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |\n",100.0*info[0][0][0],100.0*info[0][0][1],100.0*info[0][0][2],100.0*info[0][0][3],100.0*info[0][0][4]);
    fprintf(pFile,"|  Sro  | hasMat   | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |\n",100.0*info[0][1][0],100.0*info[0][1][1],100.0*info[0][1][2],100.0*info[0][1][3],100.0*info[0][1][4]);
    fprintf(pFile,"|  Sro  | hasAff   | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |\n",100.0*info[0][2][0],100.0*info[0][2][1],100.0*info[0][2][2],100.0*info[0][2][3],100.0*info[0][2][4]);
    fprintf(pFile,"|  sRo  |          | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |\n",100.0*info[1][0][0],100.0*info[1][0][1],100.0*info[1][0][2],100.0*info[1][0][3],100.0*info[1][0][4]);
    fprintf(pFile,"|  srO  | atLoc    | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |\n",100.0*info[2][0][0],100.0*info[2][0][1],100.0*info[2][0][2],100.0*info[2][0][3],100.0*info[2][0][4]);
    fprintf(pFile,"|  srO  | hasMat   | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |\n",100.0*info[2][1][0],100.0*info[2][1][1],100.0*info[2][1][2],100.0*info[2][1][3],100.0*info[2][1][4]);
    fprintf(pFile,"|  srO  | hasAff   | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |\n",100.0*info[2][2][0],100.0*info[2][2][1],100.0*info[2][2][2],100.0*info[2][2][3],100.0*info[2][2][4]);
    fclose(pFile);
}

void eval_print(const char* prefix, const vector<vector<vector<double>>>& info) {
    printf("------------------%s---------------------\n", prefix);
    printf("| Query | Relation |   MRR* | Hits10*| Hits5* | Hits1* |   MAP* |\n");
    printf("|  Sro  | atLoc    | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |\n",100.0*info[0][0][0],100.0*info[0][0][1],100.0*info[0][0][2],100.0*info[0][0][3],100.0*info[0][0][4]);
    printf("|  Sro  | hasMat   | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |\n",100.0*info[0][1][0],100.0*info[0][1][1],100.0*info[0][1][2],100.0*info[0][1][3],100.0*info[0][1][4]);
    printf("|  Sro  | hasAff   | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |\n",100.0*info[0][2][0],100.0*info[0][2][1],100.0*info[0][2][2],100.0*info[0][2][3],100.0*info[0][2][4]);
    printf("|  sRo  |          | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |\n",100.0*info[1][0][0],100.0*info[1][0][1],100.0*info[1][0][2],100.0*info[1][0][3],100.0*info[1][0][4]);
    printf("|  srO  | atLoc    | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |\n",100.0*info[2][0][0],100.0*info[2][0][1],100.0*info[2][0][2],100.0*info[2][0][3],100.0*info[2][0][4]);
    printf("|  srO  | hasMat   | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |\n",100.0*info[2][1][0],100.0*info[2][1][1],100.0*info[2][1][2],100.0*info[2][1][3],100.0*info[2][1][4]);
    printf("|  srO  | hasAff   | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |\n",100.0*info[2][2][0],100.0*info[2][2][1],100.0*info[2][2][2],100.0*info[2][2][3],100.0*info[2][2][4]);
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

void fold_sum(vector<vector<vector<double>>>& metric_total, vector<vector<vector<double>>>& metric) {
    for(unsigned i = 0; i < metric_total.size(); i++){
        for(unsigned j = 0; j < metric_total[i].size(); j++){
            for(unsigned k = 0; k < metric_total[i][j].size(); k++){
                metric_total[i][j][k] += 0.2 * metric[i][j][k];
            }
        }
    }
}

double condense_mrr(vector<vector<vector<double>>>& metric) {
    double mean_mrr = 0.0;
    for (unsigned i=0;i<metric[0].size();i++) {
        mean_mrr += (metric[0][i][0] + metric[2][i][0])/2.0;
    }
    mean_mrr = mean_mrr / metric[0].size();
    return mean_mrr;
}

class InteractiveTerminal {
    int ne;
    int nr;
    typedef tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>> ranked_list_set;

public:
    InteractiveTerminal(int ne, int nr) :
        // sets the number of relations and entities, also gives the set of
        // s,r,o used to either test, train, or validate and the complete s,r,o
        // bucket
        ne(ne), nr(nr) {}

    void query(const Model *model, const SROBucket *observed_bucket,
               unordered_map<string, int> ent_map, unordered_map<string, int> rel_map) {
        int s,r,o;
        string ss,rr,oo;
        capture_sro(s,r,o,ss,rr,oo,ent_map,rel_map);
        tuple<int,int,int> list_sizes = make_tuple(10, 3, 10);
        auto ranked_lists = this->get_ranked_list(model,observed_bucket,make_tuple(s,r,o),list_sizes);
        printf("\n\nQuery Triple: %s %s %s\n", ss.c_str(), rr.c_str(), oo.c_str());
        FILE * pFile;
        pFile = fopen("interactive_log.txt","a");
        fprintf(pFile,"\n\nQuery Triple: %s %s %s\n", ss.c_str(), rr.c_str(), oo.c_str());
        fclose(pFile);
        print_query(get<0>(ranked_lists),0,ss,rr,oo,ent_map,rel_map);
        print_query(get<1>(ranked_lists),1,ss,rr,oo,ent_map,rel_map);
        print_query(get<2>(ranked_lists),2,ss,rr,oo,ent_map,rel_map);
        log_query(get<0>(ranked_lists),0,ss,rr,oo,ent_map,rel_map);
        log_query(get<1>(ranked_lists),1,ss,rr,oo,ent_map,rel_map);
        log_query(get<2>(ranked_lists),2,ss,rr,oo,ent_map,rel_map);
    }

    bool continue_interaction() {
        string input;
        bool not_done = true;
        cout << "\n\nQuery again? (Y/n)" << endl;
        cin >> input;
        if ((input == "N") || (input == "n")) not_done = false;
        return not_done;
    }

private:
    void print_query(vector<vector<double>> results, int query_type_index, string& subject, string& relation, string& object,
                     unordered_map<string, int> ent_map, unordered_map<string, int> rel_map) {
        vector<string> query_types{"r( _ ,o)", "_ (s, o)", "r(s, _ )"};
        string query_type = query_types.at(query_type_index);
        printf("\nQuery Type: %s\n", query_type.c_str());
        printf("|           Entity           |  Score  | # Observations |\n");
        for (unsigned result_index=0;result_index<results.size();result_index++) {
            if (query_type_index == 1) {
                int rel_id = results[result_index][0];
                unordered_map<string,int>::iterator rel = find_if(rel_map.begin(),rel_map.end(),[&rel_id](unordered_map<string,int>::value_type& id){return id.second == rel_id;});
                printf("| %-26s | %7.3f | %14.0f |\n", (*rel).first.c_str(),results[result_index][1],results[result_index][2]);
            } else {
                int ent_id = results[result_index][0];
                unordered_map<string,int>::iterator ent = find_if(ent_map.begin(),ent_map.end(),[&ent_id](unordered_map<string,int>::value_type& id){return id.second == ent_id;});
                printf("| %-26s | %7.3f | %14.0f |\n", (*ent).first.c_str(),results[result_index][1],results[result_index][2]);
            }
        }
    }

    void log_query(vector<vector<double>> results, int query_type_index, string& subject, string& relation, string& object,
                     unordered_map<string, int> ent_map, unordered_map<string, int> rel_map) {
        vector<string> query_types{"r( _ ,o)", "_ (s, o)", "r(s, _ )"};
        FILE * pFile;
        pFile = fopen("interactive_log.txt","a");
        string query_type = query_types.at(query_type_index);
        fprintf(pFile,"\nQuery Type: %s\n", query_type.c_str());
        fprintf(pFile,"|           Entity           |  Score  | # Observations |\n");
        for (unsigned result_index=0;result_index<results.size();result_index++) {
            if (query_type_index == 1) {
                int rel_id = results[result_index][0];
                unordered_map<string,int>::iterator rel = find_if(rel_map.begin(),rel_map.end(),[&rel_id](unordered_map<string,int>::value_type& id){return id.second == rel_id;});
                fprintf(pFile,"| %-26s | %7.3f | %14.0f |\n", (*rel).first.c_str(),results[result_index][1],results[result_index][2]);
            } else {
                int ent_id = results[result_index][0];
                unordered_map<string,int>::iterator ent = find_if(ent_map.begin(),ent_map.end(),[&ent_id](unordered_map<string,int>::value_type& id){return id.second == ent_id;});
                fprintf(pFile,"| %-26s | %7.3f | %14.0f |\n", (*ent).first.c_str(),results[result_index][1],results[result_index][2]);
            }
        }
        fclose(pFile);
    }

    void capture_sro(int& subject, int& relation, int& object,
                     string& ssubject, string& srelation, string& sobject,
                     unordered_map<string, int>& ent_map,
                     unordered_map<string, int>& rel_map)
    {
        string input;
        bool invalid = true;
        while (invalid) {
            cout << "Please enter the subject." << endl;
            cin >> input;
            try {
                subject = ent_map.at(input);
                ssubject = input;
                invalid = false;
            } catch (const out_of_range& oor) {
                cout << input << " is not a valid subject." << endl;
            }
        }
        invalid = true;
        while (invalid) {
            cout << "Please enter the relation." << endl;
            cin >> input;
            try {
                relation = rel_map.at(input);
                srelation = input;
                invalid = false;
            } catch (const out_of_range& oor) {
                cout << input << " is not a valid relation." << endl;
            }
        }
        invalid = true;
        while (invalid) {
            cout << "Please enter the object." << endl;
            cin >> input;
            try {
                object = ent_map.at(input);
                sobject = input;
                invalid = false;
            } catch (const out_of_range& oor) {
                cout << input << " is not a valid object." << endl;
            }
        }
    }

    ranked_list_set get_ranked_list(const Model* model, const SROBucket* bucket, const triplet& sro, const tuple<int,int,int>& list_sizes) {
        // gets ranked list for robocse
        vector<vector<double>> ranked_list_s;
        vector<vector<double>> ranked_list_r;
        vector<vector<double>> ranked_list_o;

        int s = get<0>(sro);
        int r = get<1>(sro);
        int o = get<2>(sro);
        int list_size_s = get<0>(list_sizes);
        int list_size_r = get<1>(list_sizes);
        int list_size_o = get<2>(list_sizes);

        // gets ranked lists for each query type
        for (int ss = 0; ss < ne; ss++) {
            double score1,score2;
            score1 = model->score(ss, r, o);
            score2 = bucket->score(ss, r, o);
            vector<double> rank{double(ss),score1,score2};
            ranked_list_s.push_back(rank);
        }
        for (int rr = 0; rr < nr; rr++) {
            double score1,score2;
            score1 = model->score(s, rr, o);
            score2 = bucket->score(s, rr, o);
            vector<double> rank{double(rr),score1,score2};
            ranked_list_r.push_back(rank);
        }
        for (int oo = 0; oo < ne; oo++) {
            double score1,score2;
            score1 = model->score(s, r, oo);
            score2 = bucket->score(s, r, oo);
            vector<double> rank{double(oo),score1,score2};
            ranked_list_o.push_back(rank);
        }
        // sort the ranked lists
        sort(ranked_list_s.begin(),ranked_list_s.end(),[](vector<double> a, vector<double> b) {
            return (a.at(1) > b.at(1));
        });
        sort(ranked_list_r.begin(),ranked_list_r.end(),[](vector<double> a, vector<double> b) {
            return (a.at(1) > b.at(1));
        });
        sort(ranked_list_o.begin(),ranked_list_o.end(),[](vector<double> a, vector<double> b) {
            return (a.at(1) > b.at(1));
        });
        vector<vector<double>> final_s(ranked_list_s.begin(),ranked_list_s.begin()+list_size_s);
        vector<vector<double>> final_r(ranked_list_r.begin(),ranked_list_r.begin()+list_size_r);
        vector<vector<double>> final_o(ranked_list_o.begin(),ranked_list_o.begin()+list_size_o);
        return make_tuple(final_s,final_r,final_o);
    }
};

int main(int argc, char **argv) {
    // default options
    string  ddir        = "./datasets/";
    string  dataset     =  "sd_thor"; // training set
    string  experiment  =  "tg_all_0"; // training set
    int     embed_dim   =  100; // dimensionality of the embedding
    double  eta         =  0.1; // related to learning rate
    double  gamma       =  1e-3; // related to gradient
    int     neg_ratio   =  9; // number of negative examples to see
    int     num_epoch   =  500; // number of epochs
    int     num_thread  =  1; // number of threads
    int     eval_freq   =  10; // how often to evaluate while training
    string  model_path; // path to output- (train) or input- (test) model
    int     phase       =  0; // whether testing or training
    int     num_scalar  =  embed_dim / 2.0;
    int     train_size  =  0;
    string  model_type  =  "Analogy";
    // parses all the ANALOGY arguments
    int i;
    if ((i = ArgPos((char *)"-embed_dim",  argc, argv)) > 0)  embed_dim   =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-eta",        argc, argv)) > 0)  eta         =  atof(argv[i+1]);
    if ((i = ArgPos((char *)"-gamma",      argc, argv)) > 0)  gamma       =  atof(argv[i+1]);
    if ((i = ArgPos((char *)"-neg_ratio",  argc, argv)) > 0)  neg_ratio   =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-num_epoch",  argc, argv)) > 0)  num_epoch   =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-num_thread", argc, argv)) > 0)  num_thread  =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-eval_freq",  argc, argv)) > 0)  eval_freq   =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-dataset",    argc, argv)) > 0)  dataset     =  string(argv[i+1]);
    if ((i = ArgPos((char *)"-experiment", argc, argv)) > 0)  experiment  =  string(argv[i+1]);
    if ((i = ArgPos((char *)"-phase", argc, argv)) > 0)       phase       =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-num_scalar", argc, argv)) > 0)  num_scalar  =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-train_size", argc, argv)) > 0)  train_size  =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-model_type", argc, argv)) > 0)  model_type  =  string(argv[i+1]);
    num_scalar  = embed_dim / 2.0;
    // lists all the ANALOGY arguments
    printf("dataset     =  %s\n", dataset.c_str());
    printf("experiment  =  %s\n", experiment.c_str());
    printf("embed_dim   =  %d\n", embed_dim);
    printf("eta         =  %e\n", eta);
    printf("gamma       =  %e\n", gamma);
    printf("neg_ratio   =  %d\n", neg_ratio);
    printf("num_epoch   =  %d\n", num_epoch);
    printf("num_thread  =  %d\n", num_thread);
    printf("eval_freq   =  %d\n", eval_freq);
    model_path = "./trained_models/" + dataset + "_" + experiment;
    printf("model_path  =  %s\n", model_path.c_str());
    printf("num_scalar  =  %d\n", num_scalar);
    printf("train_size  =  %d\n", train_size);
    printf("model_type  =  %s\n", model_type.c_str());
    printf("phase       =  %d\n", phase);
    // gets entities & relations as strings
    vector<string> ents = read_first_column(ddir+dataset+"_entities.csv");
    vector<string> rels = read_first_column(ddir+dataset+"_relations.csv");
    // creates mapping between entities/relations and unique ID
    unordered_map<string, int> ent_map = create_id_mapping(ents);
    unordered_map<string, int> rel_map = create_id_mapping(rels);
    int ne = ent_map.size();
    int nr = rel_map.size();

    // loads the train, test, and validation triplets from each dataset
    vector<triplet> sros_tr;
    vector<triplet> sros_va;
    vector<triplet> sros_te;
    vector<triplet> sros_al;
    vector<triplet> sros_skip;

    // initializes the ANALOGY model for use
    Model *model = NULL;
    model = new Analogy(ne,nr,embed_dim,num_scalar,eta,gamma);
    assert(model != NULL);

    // checks if we are in the interactive phase of program
    if (phase == 2) {
        model->load(model_path+".model");
        string fold_name = ddir+dataset+"_"+experiment;
        sros_tr = create_sros(fold_name+"_train.csv",ent_map,rel_map);
        sros_va = create_sros(fold_name+"_valid.csv",ent_map,rel_map);
        sros_te = create_sros(fold_name+"_test.csv",ent_map,rel_map);
        sros_al.insert(sros_al.end(), sros_tr.begin(), sros_tr.end());
        sros_al.insert(sros_al.end(), sros_va.begin(), sros_va.end());
        SROBucket sro_bucket_al(sros_al);
        SROBucket *sro_bucket_all = &sro_bucket_al;
        InteractiveTerminal interface(ne, nr);
        do {
            interface.query(model,sro_bucket_all,ent_map,rel_map);
        } while (interface.continue_interaction());
        return 0;
    }

    // checks if we are in the testing phase of program
    if (phase == 1) {
        vector<vector<vector<double>>> info_test_avg;
        info_test_avg.resize(3);
        for(unsigned i = 0; i < info_test_avg.size(); i++){
            info_test_avg[i].resize(nr);
            for(unsigned j = 0; j < info_test_avg[i].size(); j++){
                info_test_avg[i][j].resize(5);
                for(unsigned k = 0; k < info_test_avg[i][j].size(); k++){
                    info_test_avg[i][j][k] = 0.0;
                }
            }
        }

        string fold_name;
        string model_fold_path;
        for (int i=0;i<5;i++) {
            fold_name = ddir+dataset+"_"+experiment+"_"+to_string(i);
            model_fold_path = model_path+"_"+to_string(i)+".model";
            sros_tr = create_sros(fold_name+"_train.csv",ent_map,rel_map);
            sros_va = create_sros(fold_name+"_valid.csv",ent_map,rel_map);
            sros_te = create_sros(fold_name+"_test.csv",ent_map,rel_map);
            sros_al.clear();
            sros_skip.clear();
            sros_al.insert(sros_al.end(), sros_tr.begin(), sros_tr.end());
            sros_al.insert(sros_al.end(), sros_va.begin(), sros_va.end());
            sros_al.insert(sros_al.end(), sros_te.begin(), sros_te.end());
            sros_skip.insert(sros_skip.end(), sros_tr.begin(), sros_tr.end());
            sros_skip.insert(sros_skip.end(), sros_va.begin(), sros_va.end());

            // creates a 'bucket' object of all s,r,o triplets, used later
            SROBucket sro_bucket_al(sros_al);
            SROBucket *sro_bucket = &sro_bucket_al;
            SROBucket sro_bucket_skip(sros_skip);
            SROBucket *skip_bucket = &sro_bucket_skip;

            Evaluator evaluator_te(ne, nr, sros_te);
            vector<vector<vector<double>>> info_test;
            if (model_type == "Random") {
                cout << "Not implemented!" << endl;
                //RandomModel* random_model = new RandomModel(ne,nr,rand());
                // TODO update random evaluation to new MRR,MAP, etc.
                //info_test = evaluator_te.evaluate(random_model,sro_bucket,skip_bucket,-1);
            } else {
                model->load(model_fold_path);
                info_test = evaluator_te.evaluate(model,sro_bucket,skip_bucket,-1);
            }
            //eval_print("FOLD EVALUATION",info_test);
            fold_sum(info_test_avg, info_test);
        }
        eval_print("TEST EVALUATION",info_test_avg);
        return 0;
    }

    // loads the train, test, and validation triplets from each dataset
    sros_tr = create_sros(ddir+dataset+"_"+experiment+"_train.csv",ent_map,rel_map);
    sros_va = create_sros(ddir+dataset+"_"+experiment+"_valid.csv",ent_map,rel_map);
    sros_te = create_sros(ddir+dataset+"_"+experiment+"_test.csv",ent_map,rel_map);
    // store all the triplets in sros_al
    sros_al.insert(sros_al.end(), sros_tr.begin(), sros_tr.end());
    sros_al.insert(sros_al.end(), sros_va.begin(), sros_va.end());
    sros_al.insert(sros_al.end(), sros_te.begin(), sros_te.end());
    sros_skip.insert(sros_skip.end(), sros_tr.begin(), sros_tr.end());
    // creates a 'bucket' object of all s,r,o triplets, used later
    SROBucket sro_bucket_al(sros_al);
    SROBucket *sro_bucket = &sro_bucket_al;
    SROBucket sro_bucket_skip(sros_skip);
    SROBucket *skip_bucket = &sro_bucket_skip;

    // evaluator for validation data
    Evaluator evaluator_va(ne, nr, sros_va);
    // evaluator for training data
    Evaluator evaluator_tr(ne, nr, sros_tr);

    // thread-specific negative samplers
    vector<NegativeSampler> neg_samplers;
    for (int tid = 0; tid < num_thread; tid++) {
        // creates a bunch of random entity/relation generators
        neg_samplers.push_back(NegativeSampler(ne, nr, rand() ^ tid,
                                               ent_map, rel_map, sros_tr) );
    }

    int N = sros_tr.size(); // N is number of examples in training set
    vector<int> pi = range(N); // pi is the range of numbers in N

    clock_t start_e;
    clock_t start_t;
    double elapse_tr;
    double elapse_ev;
    double best_mrr = 0;

    omp_set_num_threads(num_thread); // tells omp lib num of threads to use

    start_t = omp_get_wtime();
    for (int epoch = 0; epoch < num_epoch; epoch++) {
        // goes through the full number of epochs for training
        if (epoch % eval_freq == 0) {
            // evaluation
            start_e = omp_get_wtime();
            auto info_tr = evaluator_tr.evaluate(model, sro_bucket, skip_bucket, 2048);
            auto info_va = evaluator_va.evaluate(model, sro_bucket, skip_bucket, 2048);
            elapse_ev = omp_get_wtime() - start_e;
            printf("Elapse   EV %f\n", elapse_ev);
            // save the best model to disk
            double curr_mrr = condense_mrr(info_va);
            if (curr_mrr > best_mrr) {
                best_mrr = curr_mrr;
                if ( !model_path.empty() )
                    model->save(model_path+".model");
                    cout << "Model saved." << endl;
            }
            //eval_print("TRAIN EVALUATION",info_tr);
            eval_log("VALID EVALUATION",info_va);
        }
        
        // shuffles all the numbers corresponding to each example
        shuffle(pi.begin(), pi.end(), GLOBAL_GENERATOR);

        start_e = omp_get_wtime();
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
                int oo = neg_samplers[tid].random_o_entity(r);
                int ss = neg_samplers[tid].random_s_entity(r);
                int rr = neg_samplers[tid].random_relation();

                // XXX: it is empirically beneficial to carry out updates even
                // if oo == o || ss == s.
                // This might be related to regularization.
                model->train(s, r, oo, false);
                model->train(ss, r, o, false);
                model->train(s, rr, o, false);   // this improves MR slightly
            }
        }
        //printf("Epoch %03d   TR Elapse    %f\n", epoch, elapse_tr);
    }
    elapse_tr = omp_get_wtime() - start_t;
    printf("Elapse   TR %f\n", elapse_tr);

    return 0;
}

