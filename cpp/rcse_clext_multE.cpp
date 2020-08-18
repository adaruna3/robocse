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
#include <time.h>

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

unordered_map<string, int> create_id_mapping(const vector<string>& items, int start) {
    // creates a mapping between items and unique IDs
    unordered_map<string, int> map;
    for (int i = 0; i < (int) items.size(); i++)
        map[items[i]] = start+i;
    return map;
}

void create_sros(vector<triplet>& given_sros, vector<triplet>& missing_sros,
                 const string& fname,
                 const unordered_map<string, int>& ent_map,
                 const unordered_map<string, int>& rel_map,
                 const unordered_map<string, int>& m_ent_map) {
    // creates a dataset of s,r,o triplets
    ifstream ifs(fname, ios::in); // file of triplets
    // make sure dataset file is open
    assert(!ifs.fail());

    string line; // triplet line variable
    string s, r, o; // subj, obj, rel holders
    int s_id,o_id; // subj/obj id holders
    bool missing_triple_flag;
    given_sros.clear();
    missing_sros.clear();

    getline(ifs, line); // skip first line
    while (getline(ifs,line)) { // go through all lines in dataset
        stringstream ss(line);
        getline(ss,s,',');
        getline(ss,r,',');
        getline(ss,o,',');
        // check for subject and object in entity map, o.w. skips triplet
        missing_triple_flag = false;
        try {
            s_id = ent_map.at(s);
        } catch (const out_of_range& e) {
            s_id = m_ent_map.at(s);
            missing_triple_flag = true;
        }
        try {
            o_id = ent_map.at(o);
        } catch (const out_of_range& e) {
            o_id = m_ent_map.at(o);
            missing_triple_flag = true;
        }
        // add triplet to list while mapping names to unique IDs
        if (missing_triple_flag) {
            missing_sros.push_back(make_tuple(s_id,rel_map.at(r),o_id));
        } else {
            given_sros.push_back(make_tuple(s_id,rel_map.at(r),o_id));
        }
    }
    ifs.close(); // close file
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

vector<double> uniform_array(int m, double l, double h) {
    // creates an M array with random numbers uniformly distributed between
    // h and l
    vector<double> matrix;
    matrix.resize(m); // creates M dim

    for (int i = 0; i < m; i++)
        // populates the matrix with random numbers uniformly b/w h and l
        matrix[i] = (h-l)*UNIFORM(GLOBAL_GENERATOR) + l;

    return matrix;
}

vector<double> uniform_array_from_E(int m, const vector<vector<double>> E) {
    // creates an M array with random numbers uniformly distributed between
    // max and min of each embedding dimension
    vector<double> matrix;
    matrix.resize(m); // creates M dim

    for (int i = 0; i < m; i++) {
        // populates the matrix with random numbers uniformly b/w h and l
        // where h and l are max/min of all embeddings along that dimension
        double l = numeric_limits<double>::infinity();
        double h = -numeric_limits<double>::infinity();
        for (unsigned j=0; j < E.size(); j++) {
            if (E[j][i] < l)
                l = E[j][i];
            if (E[j][i] > h)
                h = E[j][i];
        }
        matrix[i] = (h-l)*UNIFORM(GLOBAL_GENERATOR) + l;
    }

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

vector<double> const_array(int m, double c) {
    // creates an M constant array initialized to c
    vector<double> matrix;
    matrix.resize(m);

    for (int i = 0; i < m; i++)
            matrix[i] = c;

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
        ofstream ofs("./trained_models/"+fname+".model", ios::out);

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
        ifstream ifs("./trained_models/"+fname+".model", ios::in);
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

    virtual ~Model(){};
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
        // initialized to constant 0.000001 (used in adagrad learning rate)
        E_g = const_matrix(ne, nh, init_e);
        // creates a matrix of size number relations by embedding dim
        // initialized to constant 0.000001 (used in adagrad learning rate)
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

    tuple<double, double, double> average_precision(ranked_list_set ground_truth_set, ranked_list_set result_set) {
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

class SimilarityInsert {

protected:
    const unsigned w2v_dim = 300;

public:
    vector<vector<double>> E; // analogy entity embeddings
    vector<vector<double>> w2v_vectors; // word2vec entity embeddings

    SimilarityInsert(vector<vector<double>> E, int num_vectors, string fname) {
        this->E = E;
        // loads a KGE model from the file fname as ifstream
        ifstream ifs(fname, ios::in);
        assert(!ifs.fail());

        w2v_vectors.resize(num_vectors);
        for (unsigned i = 0; i < w2v_vectors.size(); i++) {
            w2v_vectors[i].resize(w2v_dim);
            for (unsigned j = 0; j < w2v_dim; j++) {
                // extracts entity embeddings
                ifs >> w2v_vectors[i][j];
            }
        }

        // closes model file
        ifs.close();
    }

    vector<vector<double>> insert_entity(vector<int> entity_ids) {
        vector<vector<double>> centroids;
        int num_nearest = 4;
        vector<double> centroid;
        vector<double> scores;
        vector<double> missing_vector;
        int entity_id;
        for (unsigned i=0;i<entity_ids.size();i++) {
            entity_id = entity_ids[i];
            centroid.clear();
            scores.clear();
            missing_vector.clear();
            // get vector of entity id based on ORIGINAL IDs of entities!!
            missing_vector = w2v_vectors.at(entity_id);
            // loop through entity vectors
            for (unsigned j=0;j<w2v_vectors.size();j++) {
                // compare each entity using cosine similarity and word2vec embeddings, skipping missing IDs
                if (find(entity_ids.begin(),entity_ids.end(),j) == entity_ids.end()) {
                    vector<double> candidate_vector = w2v_vectors.at(j);
                    scores.push_back(cosine_similarity_vectors(missing_vector,candidate_vector));
                } else {
                    scores.push_back(-1.0);
                }
            }
            // select the top num_nearest entities to help insert embedding
            auto nearest = sort_indexes(scores);
            auto first = nearest.end()-num_nearest;
            auto last = nearest.end();
            vector<double> candidates(first,last);
            candidates = map_ids_from_w2v(candidates, entity_ids);
            // return centroid of num_nearest most similar analogy embeddings
            centroid.resize(E[0].size());
            for (unsigned j=0;j<centroid.size();j++)
                centroid[j] = 0.0;
            for (unsigned j=0;j<candidates.size();j++)
                for (unsigned k=0;k<centroid.size();k++)
                    centroid[k] += E[candidates[j]][k];
            for (unsigned j=0;j<centroid.size();j++)
                    centroid[j] = centroid[j] / candidates.size();
            centroids.push_back(centroid);
        }
        return centroids;
    }

    vector<double> map_ids_from_w2v(vector<double> w2v_ids, vector<int> missing_ids) {
        vector<double> analogy_ids;
        int num_shifts;
        for (unsigned j=0;j<w2v_ids.size();j++) {
            num_shifts = 0;
            for (unsigned i=0;i<missing_ids.size();i++) {
                if (w2v_ids[j] > missing_ids[i]) {
                    num_shifts += 1;
                }
            }
            analogy_ids.push_back(w2v_ids[j]-num_shifts);
        }
        return analogy_ids;
    }

    template <typename T>
    vector<size_t> sort_indexes(const vector<T> &v) {
      // initialize original index locations
      vector<size_t> idx(v.size());
      iota(idx.begin(), idx.end(), 0);

      // sort indexes based on comparing values in v
      sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

      return idx;
    }

    double cosine_similarity_vectors(std::vector<double> A, std::vector<double>B) {
        double mul = 0.0;
        double d_a = 0.0;
        double d_b = 0.0 ;

        std::vector<double>::iterator B_iter = B.begin();
        std::vector<double>::iterator A_iter = A.begin();
        for( ; A_iter != A.end(); A_iter++ , B_iter++ ) {
            mul += *A_iter * *B_iter;
            d_a += *A_iter * *A_iter;
            d_b += *B_iter * *B_iter;
        }

        if (d_a == 0.0f || d_b == 0.0f) {
            throw std::logic_error(
                    "cosine similarity is not defined whenever one or both "
                    "input vectors are zero-vectors.");
        }

        return mul / sqrt(d_a * d_b);
    }
};

class Similarity2Insert {

protected:
    const unsigned w2v_dim = 300;

public:
    vector<vector<double>> E; // analogy entity embeddings
    vector<vector<double>> w2v_vectors; // word2vec entity embeddings

    Similarity2Insert(vector<vector<double>> E, int num_vectors, string fname) {
        this->E = E;
        // loads a KGE model from the file fname as ifstream
        ifstream ifs(fname, ios::in);
        assert(!ifs.fail());

        w2v_vectors.resize(num_vectors);
        for (unsigned i = 0; i < w2v_vectors.size(); i++) {
            w2v_vectors[i].resize(w2v_dim);
            for (unsigned j = 0; j < w2v_dim; j++) {
                // extracts entity embeddings
                ifs >> w2v_vectors[i][j];
            }
        }

        // closes model file
        ifs.close();
    }

    vector<vector<double>> insert_entity(vector<int> entity_ids) {
        vector<vector<double>> centroids;
        int num_nearest = 30;
        vector<double> centroid;
        vector<double> scores;
        vector<double> missing_vector;
        int entity_id;
        for (unsigned i=0;i<entity_ids.size();i++) {
            entity_id = entity_ids[i];
            centroid.clear();
            scores.clear();
            missing_vector.clear();
            // get vector of entity id based on ORIGINAL IDs of entities!!
            missing_vector = w2v_vectors.at(entity_id);
            // loop through entity vectors
            for (unsigned j=0;j<w2v_vectors.size();j++) {
                // compare each entity using cosine similarity and word2vec embeddings, skipping missing IDs
                if (find(entity_ids.begin(),entity_ids.end(),j) == entity_ids.end()) {
                    vector<double> candidate_vector = w2v_vectors.at(j);
                    scores.push_back(cosine_similarity_vectors(missing_vector,candidate_vector));
                } else {
                    scores.push_back(-1.0);
                }
            }
            // select the top num_nearest entities to help insert embedding
            auto nearest = sort_indexes(scores);
            auto first = nearest.end()-num_nearest;
            auto last = nearest.end();
            vector<double> candidates(first,last);

            // todo debug
            vector<double> weights;
            double weights_normalizer = 0;
            for (unsigned j=0;j<candidates.size();j++) {
                double weight = scores[candidates[j]];
                weights.push_back(weight);
                weights_normalizer += weight;
            }
            for (unsigned j=0;j<weights.size();j++) {
                weights[j] = weights[j] / weights_normalizer;
            }

            candidates = map_ids_from_w2v(candidates, entity_ids);

            // initialize weighted centroid of num_nearest most similar analogy embeddings
            centroid.resize(E[0].size());
            for (unsigned j=0;j<centroid.size();j++)
                centroid[j] = 0.0;

            // calculate the weighted centroid todo debug
            for (unsigned j=0;j<candidates.size();j++)
                for (unsigned k=0;k<centroid.size();k++)
                    centroid[k] += E[candidates[j]][k] * weights[j];

            centroids.push_back(centroid);
        }
        return centroids;
    }

    vector<double> map_ids_from_w2v(vector<double> w2v_ids, vector<int> missing_ids) {
        vector<double> analogy_ids;
        int num_shifts;
        for (unsigned j=0;j<w2v_ids.size();j++) {
            num_shifts = 0;
            for (unsigned i=0;i<missing_ids.size();i++) {
                if (w2v_ids[j] > missing_ids[i]) {
                    num_shifts += 1;
                }
            }
            analogy_ids.push_back(w2v_ids[j]-num_shifts);
        }
        return analogy_ids;
    }

    template <typename T>
    vector<size_t> sort_indexes(const vector<T> &v) {
      // initialize original index locations
      vector<size_t> idx(v.size());
      iota(idx.begin(), idx.end(), 0);

      // sort indexes based on comparing values in v
      sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

      return idx;
    }

    double cosine_similarity_vectors(std::vector<double> A, std::vector<double>B) {
        double mul = 0.0;
        double d_a = 0.0;
        double d_b = 0.0 ;

        std::vector<double>::iterator B_iter = B.begin();
        std::vector<double>::iterator A_iter = A.begin();
        for( ; A_iter != A.end(); A_iter++ , B_iter++ ) {
            mul += *A_iter * *B_iter;
            d_a += *A_iter * *A_iter;
            d_b += *B_iter * *B_iter;
        }

        if (d_a == 0.0f || d_b == 0.0f) {
            throw std::logic_error(
                    "cosine similarity is not defined whenever one or both "
                    "input vectors are zero-vectors.");
        }

        return mul / sqrt(d_a * d_b);
    }
};

class RelationalNodeInsert {

protected:

public:
    vector<vector<double>> E; // analogy entity embeddings
    vector<vector<double>> R; // analogy relation embeddings
    int nh1;
    int nh2;
    vector<triplet> unique_triples;

    RelationalNodeInsert(vector<vector<double>> E, vector<vector<double>> R, const vector<triplet>& sros,
                     int embedding_dim, int num_scalar) {
        this->E = E;
        this->R = R;
        this->nh1 = num_scalar;
        this->nh2 = embedding_dim-num_scalar;
        for (auto new_triplet : sros)
            if (find(unique_triples.begin(),unique_triples.end(),new_triplet) == unique_triples.end())
                this->unique_triples.push_back(new_triplet);
    }

    vector<vector<double>> insert_entity(vector<int> entity_ids) {
        vector<vector<double>> centroids;
        int entity_id;
        vector<double> scores;
        vector<int> relation_types;
        relation_types.resize(R.size());
        for (unsigned j=0;j<relation_types.size();j++)
            relation_types[j] = j;
        for (unsigned entity_id_idx=0;entity_id_idx<entity_ids.size();entity_id_idx++) {
            entity_id = entity_ids[entity_id_idx];
            scores.clear();
            scores.resize(E.size());
            for (unsigned j=0;j<scores.size();j++)
                scores[j] = 0.0;
            for (auto relation_type : relation_types) { // begin averaging resultant vectors
                // initialize the centroid to 0
                vector<double> centroid;
                centroid.resize(nh1+nh2);
                for(unsigned j=0;j<centroid.size();j++)
                    centroid[j] = 0.0;
                double divisor = 0.0;
                for (auto unique_triple : unique_triples) {
                    int r = get<1>(unique_triple);
                    int s = get<0>(unique_triple);
                    int o = get<2>(unique_triple);
                    bool invalid_status = invalid_triple(s, r, o, relation_type, entity_id, entity_ids);
                    if (invalid_status)
                        continue;
                    if (s == entity_id) { // compute right-sided resultant
                        int i = 0;
                        int nh2_2 = nh2/2;
                        for (;i<nh1;i++)
                            centroid[i] += (1/R[r][i]) * E[o][i];
                        for (;i<nh1+nh2_2;i++) {
                            double det_block = R[r][i]*R[r][i]+R[r][nh2_2+i]*R[r][nh2_2+i];
                            centroid[nh2_2+i] += (1/det_block) * (R[r][i]*E[o][nh2_2+i] - R[r][nh2_2+i]*E[o][i]);
                            centroid[i] += (1/det_block) * (R[r][i]*E[o][i] + R[r][nh2_2+i]*E[o][nh2_2+i]);
                        }
                    } else if (o == entity_id) { // compute left-sided resultant
                        int i = 0;
                        int nh2_2 = nh2/2;
                        for (;i<nh1;i++)
                            centroid[i] += E[s][i]*R[r][i];
                        for (;i<nh1+nh2_2;i++) {
                            centroid[nh2_2+i] += E[s][nh2_2+i]*R[r][i] + E[s][i]*R[r][nh2_2+i];
                            centroid[i] += E[s][i]*R[r][i] - E[s][nh2_2+i]*R[r][nh2_2+i];
                        }
                    } else {
                        cout << "Error with entity ID and insert triples." << endl;
                        exit(0);
                    }
                    divisor += 1.0;
                }
                if (divisor == 0.0) // skip when no triples for relation
                    continue;
                // averages values to get centroid
                for (unsigned j=0;j<centroid.size();j++)
                    centroid[j] = centroid[j] / divisor;
                // scores entities based on cosine similarity
                for (unsigned j=0;j<E.size();j++) {
                    scores[j] += cosine_similarity_vectors(centroid,E[j]);
                }
            }
            // select the top num_nearest entities to help insert embedding
            int num_nearest = 4;
            auto nearest = sort_indexes(scores);
            auto first = nearest.end()-num_nearest;
            auto last = nearest.end();
            vector<double> candidates(first,last);
            // return centroid of num_nearest most similar analogy embeddings
            vector<double> centroid;
            centroid.resize(nh1+nh2);
            for (unsigned i=0;i<centroid.size();i++)
                centroid[i] = 0.0;
            for (unsigned i = 0; i < candidates.size(); i++)
                for (unsigned j = 0; j < centroid.size(); j++)
                    centroid[j] += E[candidates[i]][j];
            for (unsigned j = 0; j < centroid.size(); j++)
                    centroid[j] = centroid[j] / candidates.size();
            centroids.push_back(centroid);
        }
        return centroids;
    }

    bool invalid_triple(int s, int r, int o, int r_type, int missing_entity, vector<int> missing) {
        if (r != r_type)
            return true; // skip bc triple not correct relation
        if  ( (s != missing_entity) && (o != missing_entity) )
            return true; // skip bc triple not related to missing entity
        if (s == missing_entity) {
            if ( find(missing.begin(),missing.end(),o) != missing.end() ) {
                return true; // skip bc triple involves other missing entity
            }
        }
        if (o == missing_entity) {
            if ( find(missing.begin(),missing.end(),s) != missing.end() ) {
                return true; // skip bc triple involves other missing entity
            }
        }
        return false;
    }

    template <typename T>
    vector<size_t> sort_indexes(const vector<T> &v) {
      // initialize original index locations
      vector<size_t> idx(v.size());
      iota(idx.begin(), idx.end(), 0);

      // sort indexes based on comparing values in v
      sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

      return idx;
    }

    double cosine_similarity_vectors(std::vector<double> A, std::vector<double>B) {
        double mul = 0.0;
        double d_a = 0.0;
        double d_b = 0.0 ;

        std::vector<double>::iterator B_iter = B.begin();
        std::vector<double>::iterator A_iter = A.begin();
        for( ; A_iter != A.end(); A_iter++ , B_iter++ ) {
            mul += *A_iter * *B_iter;
            d_a += *A_iter * *A_iter;
            d_b += *B_iter * *B_iter;
        }

        if (d_a == 0.0f || d_b == 0.0f) {
            throw std::logic_error(
                    "cosine similarity is not defined whenever one or both "
                    "input vectors are zero-vectors.");
        }

        return mul / sqrt(d_a * d_b);
    }
};

class HybridInsert {

protected:
    const unsigned w2v_dim = 300;

public:
    vector<vector<double>> E; // analogy entity embeddings
    vector<vector<double>> R; // analogy relation embeddings
    vector<vector<double>> w2v_vectors; // word2vec entity embeddings
    int nh1;
    int nh2;
    vector<triplet> unique_triples;

    HybridInsert(vector<vector<double>> E, vector<vector<double>> R, const vector<triplet>& sros,
                 int embedding_dim, int num_scalar, int num_vectors, string w2v_fname) {
        this->E = E;
        this->R = R;
        this->nh1 = num_scalar;
        this->nh2 = embedding_dim-num_scalar;
        // loads insert triples
        for (auto new_triplet : sros)
            if (find(unique_triples.begin(),unique_triples.end(),new_triplet) == unique_triples.end())
                this->unique_triples.push_back(new_triplet);
        // loads w2v embeddings
        ifstream ifs(w2v_fname, ios::in);
        assert(!ifs.fail());
        w2v_vectors.resize(num_vectors);
        for (unsigned i = 0; i < w2v_vectors.size(); i++) {
            w2v_vectors[i].resize(w2v_dim);
            for (unsigned j = 0; j < w2v_dim; j++) {
                ifs >> w2v_vectors[i][j];
            }
        }
        ifs.close();
    }

    vector<vector<double>> insert_entity(vector<int> w2v_ids_missing, vector<int> analogy_ids_missing) {
        vector<vector<double>> centroids;
        int analogy_id_missing;
        vector<double> scores;
        vector<int> relation_types;
        relation_types.resize(R.size());
        for (unsigned j=0;j<relation_types.size();j++)
            relation_types[j] = j;
        for (unsigned entity_id_idx=0;entity_id_idx<w2v_ids_missing.size();entity_id_idx++) {
            analogy_id_missing = analogy_ids_missing[entity_id_idx];
            vector<double> candidates = get_candidates(w2v_ids_missing[entity_id_idx],w2v_ids_missing);
            scores.clear();
            scores.resize(candidates.size());
            for (unsigned j=0;j<scores.size();j++)
                scores[j] = 0.0;
            for (auto relation_type : relation_types) { // begin averaging resultant vectors
                // intialize the centroid to 0
                vector<double> centroid;
                centroid.resize(nh1+nh2);
                for(unsigned j=0;j<centroid.size();j++)
                    centroid[j] = 0.0;
                double divisor = 0.0;
                for (auto unique_triple : unique_triples) {
                    int r = get<1>(unique_triple);
                    int s = get<0>(unique_triple);
                    int o = get<2>(unique_triple);
                    bool invalid_status = invalid_triple(s, r, o, relation_type, analogy_id_missing, analogy_ids_missing);
                    if (invalid_status)
                        continue;
                    if (s == analogy_id_missing) { // compute right-sided resultant
                        int i = 0;
                        int nh2_2 = nh2/2;
                        for (;i<nh1;i++)
                            centroid[i] += (1/R[r][i]) * E[o][i];
                        for (;i<nh1+nh2_2;i++) {
                            double det_block = R[r][i]*R[r][i]+R[r][nh2_2+i]*R[r][nh2_2+i];
                            centroid[nh2_2+i] += (1/det_block) * (R[r][i]*E[o][nh2_2+i] - R[r][nh2_2+i]*E[o][i]);
                            centroid[i] += (1/det_block) * (R[r][i]*E[o][i] + R[r][nh2_2+i]*E[o][nh2_2+i]);
                        }
                    } else if (o == analogy_id_missing) { // compute left-sided resultant
                        int i = 0;
                        int nh2_2 = nh2/2;
                        for (;i<nh1;i++)
                            centroid[i] += E[s][i]*R[r][i];
                        for (;i<nh1+nh2_2;i++) {
                            centroid[nh2_2+i] += E[s][nh2_2+i]*R[r][i] + E[s][i]*R[r][nh2_2+i];
                            centroid[i] += E[s][i]*R[r][i] - E[s][nh2_2+i]*R[r][nh2_2+i];
                        }
                    } else {
                        cout << "Error with entity ID and insert triples." << endl;
                        exit(0);
                    }
                    divisor += 1.0;
                }
                if (divisor == 0.0) // skip when no triples for relation
                    continue;
                // averages values to get centroid
                for (unsigned j=0;j<centroid.size();j++)
                    centroid[j] = centroid[j] / divisor;
                // scores candidates based on cosine similarity
                for (unsigned j=0;j<candidates.size();j++) {
                    scores[j] += cosine_similarity_vectors(centroid,E[candidates[j]]);
                }
            }

            // select the top num_nearest entities to help insert embedding
            int num_nearest = 4;
            auto nearest = sort_indexes(scores);
            auto first = nearest.end()-num_nearest;
            auto last = nearest.end();
            vector<double> final_candidates(first,last);
            // get ids
            for (unsigned i=0;i<final_candidates.size();i++) {
                final_candidates[i] = candidates[final_candidates[i]];
            }
            // return centroid of num_nearest most similar analogy embeddings
            vector<double> centroid;
            centroid.resize(nh1+nh2);
            for (unsigned i=0;i<centroid.size();i++)
                centroid[i] = 0.0;
            for (unsigned i = 0; i < final_candidates.size(); i++)
                for (unsigned j = 0; j < centroid.size(); j++)
                    centroid[j] += E[final_candidates[i]][j];
            for (unsigned j = 0; j < centroid.size(); j++)
                    centroid[j] = centroid[j] / final_candidates.size();
            centroids.push_back(centroid);
        }
        return centroids;
    }

    bool invalid_triple(int s, int r, int o, int r_type, int missing_entity, vector<int> missing) {
        if (r != r_type)
            return true; // skip bc triple not correct relation
        if ( (s != missing_entity) && (o != missing_entity) )
            return true; // skip bc triple not related to missing entity
        if (s == missing_entity) {
            if ( find(missing.begin(),missing.end(),o) != missing.end() ) {
                return true; // skip bc triple involves other missing entity
            }
        }
        if (o == missing_entity) {
            if ( find(missing.begin(),missing.end(),s) != missing.end() ) {
                return true; // skip bc triple involves other missing entity
            }
        }
        return false;
    }

    vector<double> get_candidates(int w2v_id_missing, vector<int> entity_ids) {
        int num_nearest = 10;
        vector<double> scores;
        // get vector of entity id based on ORIGINAL IDs of entities!!
        vector<double> missing_vector = w2v_vectors.at(w2v_id_missing);
        // loop through w2v entity vectors
        for (unsigned i=0;i<w2v_vectors.size();i++) {
            // compare each entity using cosine similarity and word2vec embeddings
            if (find(entity_ids.begin(),entity_ids.end(),i) == entity_ids.end()) {
                vector<double> candidate_vector = w2v_vectors.at(i);
                scores.push_back(cosine_similarity_vectors(missing_vector,candidate_vector));
            } else {
                scores.push_back(-1.0);
            }
        }
        // select the top num_nearest entities to help insert embedding
        auto nearest = sort_indexes(scores);
        auto first = nearest.end()-num_nearest;
        auto last = nearest.end();
        vector<double> candidates(first,last);
        candidates = map_ids_from_w2v(candidates,entity_ids);
        return candidates;
    }

    /*vector<double> insert_entity(int w2v_id_missing, int analogy_id_missing) {
        vector<double> centroid;
        int num_nearest = 4;
        vector<double> scores;
        // gets candidate ids according to relational_node insert
        vector<double> candidates = get_candidates(analogy_id_missing);
        candidates = map_ids_to_w2v(candidates,w2v_id_missing);
        // get vector of entity id based on ORIGINAL IDs of entities!!
        vector<double> missing_vector = w2v_vectors.at(w2v_id_missing);
        // loop through w2v entity vectors
        for (unsigned i=0;i<candidates.size();i++) {
            // compare each entity using cosine similarity and word2vec embeddings
            vector<double> candidate_vector = w2v_vectors.at(candidates[i]);
            scores.push_back(cosine_similarity_vectors(missing_vector,candidate_vector));
        }
        // select the top num_nearest entities to help insert embedding
        auto nearest = sort_indexes(scores);
        auto first = nearest.end()-num_nearest;
        auto last = nearest.end();
        vector<double> final_candidates(first,last);
        // get ids
        for (unsigned i=0;i<final_candidates.size();i++) {
            final_candidates[i] = candidates[final_candidates[i]];
        }
        final_candidates = map_ids_from_w2v(final_candidates,w2v_id_missing);
        // return centroid of num_nearest most similar analogy embeddings
        centroid.resize(E[0].size());
        for (unsigned i=0;i<centroid.size();i++)
            centroid[i] = 0.0;
        for (unsigned i = 0; i < final_candidates.size(); i++)
            for (unsigned j = 0; j < centroid.size(); j++)
                centroid[j] += E[final_candidates[i]][j];
        for (unsigned j = 0; j < centroid.size(); j++)
                centroid[j] = centroid[j] / final_candidates.size();
        return centroid;
    }

    vector<double> get_candidates(int entity_id) {
        vector<double> scores;
        scores.resize(E.size());
        for (unsigned j=0;j<scores.size();j++)
            scores[j] = 0.0;
        vector<int> relation_types;
        relation_types.resize(R.size());
        for (unsigned j=0;j<relation_types.size();j++)
            relation_types[j] = j;
        for (auto relation_type : relation_types) { // begin averaging resultant vectors
            // intialize the centroid to 0
            vector<double> centroid;
            centroid.resize(nh1+nh2);
            for(unsigned j=0;j<centroid.size();j++)
                centroid[j] = 0.0;
            double divisor = 0.0;
            for (auto unique_triple : unique_triples) {
                int r = get<1>(unique_triple);
                if (r != relation_type) // skip so centroid is all same relation
                    continue;
                int s = get<0>(unique_triple);
                int o = get<2>(unique_triple);
                if (s == entity_id) { // compute right-sided resultant
                    int i = 0;
                    int nh2_2 = nh2/2;
                    for (;i<nh1;i++)
                        centroid[i] += (1/R[r][i]) * E[o][i];
                    for (;i<nh1+nh2_2;i++) {
                        double det_block = R[r][i]*R[r][i]+R[r][nh2_2+i]*R[r][nh2_2+i];
                        centroid[nh2_2+i] += (1/det_block) * (R[r][i]*E[o][nh2_2+i] - R[r][nh2_2+i]*E[o][i]);
                        centroid[i] += (1/det_block) * (R[r][i]*E[o][i] + R[r][nh2_2+i]*E[o][nh2_2+i]);
                    }
                } else if (o == entity_id) { // compute left-sided resultant
                    int i = 0;
                    int nh2_2 = nh2/2;
                    for (;i<nh1;i++)
                        centroid[i] += E[s][i]*R[r][i];
                    for (;i<nh1+nh2_2;i++) {
                        centroid[nh2_2+i] += E[s][nh2_2+i]*R[r][i] + E[s][i]*R[r][nh2_2+i];
                        centroid[i] += E[s][i]*R[r][i] - E[s][nh2_2+i]*R[r][nh2_2+i];
                    }
                } else {
                    cout << "Error with entity ID and insert triples." << endl;
                    exit(0);
                }
                divisor += 1.0;
            }
            if (divisor == 0.0) // skip when no triples for relation
                continue;
            // averages values to get centroid
            for (unsigned j=0;j<centroid.size();j++)
                centroid[j] = centroid[j] / divisor;
            // scores entities based on cosine similarity
            for (unsigned j=0;j<E.size();j++) {
                scores[j] += cosine_similarity_vectors(centroid,E[j]);
            }
        }
        // select the top num_nearest entities to help insert embedding
        int num_nearest = 10;
        auto nearest = sort_indexes(scores);
        auto first = nearest.end()-num_nearest;
        auto last = nearest.end();
        vector<double> candidates(first,last);
        return candidates;
    }*/

    vector<double> map_ids_to_w2v(vector<double> analogy_ids, int missing_id) {
        vector<double> w2v_ids;
        for (unsigned j=0;j<analogy_ids.size();j++) {
            if (analogy_ids[j] >= missing_id) {
                w2v_ids.push_back(analogy_ids[j]+1);
            } else {
                w2v_ids.push_back(analogy_ids[j]);
            }
        }
        return w2v_ids;
    }

    vector<double> map_ids_from_w2v(vector<double> w2v_ids, vector<int> missing_ids) {
        vector<double> analogy_ids;
        int num_shifts;
        for (unsigned j=0;j<w2v_ids.size();j++) {
            num_shifts = 0;
            for (unsigned i=0;i<missing_ids.size();i++) {
                if (w2v_ids[j] > missing_ids[i]) {
                    num_shifts += 1;
                }
            }
            analogy_ids.push_back(w2v_ids[j]-num_shifts);
        }
        return analogy_ids;
    }

    template <typename T>
    vector<size_t> sort_indexes(const vector<T> &v) {
      // initialize original index locations
      vector<size_t> idx(v.size());
      iota(idx.begin(), idx.end(), 0);

      // sort indexes based on comparing values in v
      sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

      return idx;
    }

    double cosine_similarity_vectors(std::vector<double> A, std::vector<double>B) {
        double mul = 0.0;
        double d_a = 0.0;
        double d_b = 0.0 ;

        std::vector<double>::iterator B_iter = B.begin();
        std::vector<double>::iterator A_iter = A.begin();
        for( ; A_iter != A.end(); A_iter++ , B_iter++ ) {
            mul += *A_iter * *B_iter;
            d_a += *A_iter * *A_iter;
            d_b += *B_iter * *B_iter;
        }

        if (d_a == 0.0f || d_b == 0.0f) {
            throw std::logic_error(
                    "cosine similarity is not defined whenever one or both "
                    "input vectors are zero-vectors.");
        }

        return mul / sqrt(d_a * d_b);
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

void eval_csv(string fname, const vector<vector<vector<double>>>& info) {
    FILE * pFile;
    pFile = fopen(fname.c_str(),"w");
    fprintf(pFile,"%.2f,%.2f,%.2f,%.2f,%.2f\n",100.0*info[0][0][0],100.0*info[0][0][1],100.0*info[0][0][2],100.0*info[0][0][3],100.0*info[0][0][4]);
    fprintf(pFile,"%.2f,%.2f,%.2f,%.2f,%.2f\n",100.0*info[0][1][0],100.0*info[0][1][1],100.0*info[0][1][2],100.0*info[0][1][3],100.0*info[0][1][4]);
    fprintf(pFile,"%.2f,%.2f,%.2f,%.2f,%.2f\n",100.0*info[0][2][0],100.0*info[0][2][1],100.0*info[0][2][2],100.0*info[0][2][3],100.0*info[0][2][4]);
    fprintf(pFile,"%.2f,%.2f,%.2f,%.2f,%.2f\n",100.0*info[1][0][0],100.0*info[1][0][1],100.0*info[1][0][2],100.0*info[1][0][3],100.0*info[1][0][4]);
    fprintf(pFile,"%.2f,%.2f,%.2f,%.2f,%.2f\n",100.0*info[2][0][0],100.0*info[2][0][1],100.0*info[2][0][2],100.0*info[2][0][3],100.0*info[2][0][4]);
    fprintf(pFile,"%.2f,%.2f,%.2f,%.2f,%.2f\n",100.0*info[2][1][0],100.0*info[2][1][1],100.0*info[2][1][2],100.0*info[2][1][3],100.0*info[2][1][4]);
    fprintf(pFile,"%.2f,%.2f,%.2f,%.2f,%.2f\n",100.0*info[2][2][0],100.0*info[2][2][1],100.0*info[2][2][2],100.0*info[2][2][3],100.0*info[2][2][4]);
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

class Logger {

protected:
    vector<vector<double>> log_values;

public:
    Logger() { }

    void new_sample(vector<double> sample) {
        log_values.push_back(sample);
    }

    void flush_log(string& fname) {
        ofstream ofs("./robocse_logging/"+fname+".csv", ios::out);
        for (unsigned i = 0; i < log_values.size(); i++) {
            for (unsigned j = 0; j < log_values[i].size(); j++)
                ofs << log_values[i][j] << ',';
            ofs << endl;
        }
        ofs.close();
    }
};

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

vector<triplet> select_insert_triples(vector<triplet>& missing_triples, vector<int> missing_ent_ids) {
    vector<triplet> unique_triples;
    vector<triplet> unique_insert_triples;
    int s;
    int o;
    int missing_entity;
    // get unique triples
    for (unsigned j=0;j<missing_ent_ids.size();j++) {
        unique_triples.clear();
        missing_entity = missing_ent_ids[j];
        for (auto new_triplet : missing_triples) {
            s = get<0>(new_triplet);
            o = get<2>(new_triplet);
            if ( (s != missing_entity) && (o != missing_entity) )
                continue;
            if (find(unique_triples.begin(),unique_triples.end(),new_triplet) == unique_triples.end())
                unique_triples.push_back(new_triplet);
        }
        // shuffle and select x% for insertion
        random_shuffle(unique_triples.begin(),unique_triples.end());
        unsigned num_insert_triples = (int) round((50.0 / 100) * (float) unique_triples.size() );
        unique_insert_triples.insert(unique_insert_triples.end(),unique_triples.begin(),unique_triples.begin()+num_insert_triples);
    }

    vector<triplet> insert_triples;
    for (unsigned i=0;i<missing_triples.size();i++) {
        triplet new_triplet = missing_triples[i];
        if (find(unique_insert_triples.begin(),unique_insert_triples.end(),new_triplet) != unique_insert_triples.end()) {
            insert_triples.push_back(new_triplet);
            missing_triples.erase(missing_triples.begin()+i);
        }
    }
    return insert_triples;
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
    // gets the weighted average mrr
    double mean_mrr = 0.0;
    double divisor = 0;
    double counts[4] = {4312.0,2963.0,7484.0,14759.0};
    for (unsigned i=0;i<metric[0].size();i++) {
        // checks for NANs
        if ( (std::isnan(metric[0][i][0]) == 0) && (std::isnan(metric[2][i][0]) == 0) ) {
            mean_mrr += counts[i]*((metric[0][i][0]+metric[2][i][0])/2.0);
            divisor += counts[i];
        }
    }
    if (std::isnan(metric[1][0][0]) == 0) {
        mean_mrr += counts[3]*metric[1][0][0];
        divisor += counts[3];
    }
    mean_mrr = mean_mrr / divisor;
    return mean_mrr;
}

double get_batch_learn_performance(Evaluator& batch_evaluator, SROBucket* all_bucket,
                                   SROBucket* skip_bucket, string model_path_root,
                                   int truncate, vector<int> missing_ent_ids, int num_missing,
                                   int missing_set, string performance_type) {
    unsigned max_model_number = 5;
    double avg_mrr = 0;
    int num_shifts;
    string fname = "./trained_models/batch_learn/"+model_path_root+"_"+to_string(num_missing)+"_"+to_string(missing_set)+"_"+performance_type+".txt";

    // checks if value available
    ifstream ifs(fname, ios::in);
    if (ifs.good()) {
        ifs >> avg_mrr;
        ifs.close();
    } else { // otherwise, computes value
        cout << "Calculating batch learning performance..." << endl;
        for (unsigned model_number=0;model_number<max_model_number;model_number++) {
            string model_path = "batch_learn/" + model_path_root + "_" + to_string(model_number);
            Model* batch_model = new Analogy(106,3,100,50,0.1,1e-3);
            assert(batch_model != NULL);
            batch_model->load(model_path);
            // modify the model to account for switched entity ordering
            for (unsigned missing_ent_idx=0;missing_ent_idx<missing_ent_ids.size();missing_ent_idx++) {
                int missing_ent_id = missing_ent_ids[missing_ent_idx];
                num_shifts = 0;
                for (unsigned i=0;i<(unsigned) missing_ent_idx;i++) {
                    if (missing_ent_id > missing_ent_ids[i]) {
                        num_shifts += 1;
                    }
                }
                vector<double> switch_vector(batch_model->E[missing_ent_id-num_shifts].begin(),batch_model->E[missing_ent_id-num_shifts].end());
                batch_model->E.erase(batch_model->E.begin()+missing_ent_id-num_shifts);
                batch_model->E.push_back(switch_vector);
            }
            auto info = batch_evaluator.evaluate(batch_model, all_bucket, skip_bucket, truncate);
            avg_mrr += condense_mrr(info);
            delete batch_model;
        }
        avg_mrr = avg_mrr / max_model_number;
        // stores computed value
        ofstream ofs(fname, ios::out);
        ofs << avg_mrr << endl;
        ofs.close();
    }
    return avg_mrr;
}

void initialize_info(vector<vector<vector<double>>>& info, int nr) {
    info.resize(3);
    for(unsigned i = 0; i < info.size(); i++){
        info[i].resize(nr);
        for(unsigned j = 0; j < info[i].size(); j++){
            info[i][j].resize(5);
            for(unsigned k = 0; k < info[i][j].size(); k++){
                info[i][j][k] = 0.0;
            }
        }
    }
}

void sum_info(vector<vector<vector<double>>>& info_avg,
              vector<vector<vector<double>>>& info,
              vector<vector<vector<double>>>& totals) {
    for(unsigned i = 0; i < info.size(); i++){
        for(unsigned j = 0; j < info[i].size(); j++){
            for(unsigned k = 0; k < info[i][j].size(); k++){
                if (std::isnan(info[i][j][k]) == 0) {
                    info_avg[i][j][k] += info[i][j][k];
                    totals[i][j][k] += 1;
                }
            }
        }
    }
}

void average_totals(vector<vector<vector<double>>>& info,
                    vector<vector<vector<double>>>& totals) {
    for(unsigned i = 0; i < info.size(); i++)
        for(unsigned j = 0; j < info[i].size(); j++)
            for(unsigned k = 0; k < info[i][j].size(); k++)
                    info[i][j][k] = info[i][j][k] / totals[i][j][k];
}

bool model_exists(string& model_path) {
    ifstream ifs("./trained_models/"+model_path+".model", ios::in);
    return ifs.good();
}

vector<string> get_missing_ent_files(int num_ents, int num_samples, int num_sets, string fileroot) {
    vector<string> missing_ent_files;
    auto seed_time = time(NULL);
    default_random_engine generator = default_random_engine(seed_time);
    uniform_int_distribution<int> unif_e = uniform_int_distribution<int>(0, num_ents-1);

    if (fileroot == "sd_thor_mp3d_tg_all_0") { // check for matterport3d missing ent sampling
        unif_e = uniform_int_distribution<int>(106, num_ents-1);
    }

    for (unsigned i=0;i<(unsigned) num_sets;i++) {
        string filepath = "./trained_models/missing_ent_files/"+fileroot+"_ments_"+to_string(num_samples)+"_"+to_string(i)+".csv";
        ifstream ifs(filepath, ios::in);
        if (ifs.good()){
            ifs.close();
            missing_ent_files.push_back(filepath);
        } else {
            ofstream ofs(filepath, ios::out);
            int new_value;
            vector<int> missing_ids;
            missing_ids.resize(0);
            while(missing_ids.size() < (unsigned) num_samples) {
                new_value = unif_e(generator);
                if (find(missing_ids.begin(),missing_ids.end(),new_value) == missing_ids.end()) {
                    missing_ids.push_back(new_value);
                    ofs << new_value << endl;
                }
            }
            ofs.close();
            missing_ent_files.push_back(filepath);
        }
    }
    return missing_ent_files;
}

vector<int> get_missing_ent_ids(string filepath) {
    ifstream ifs(filepath, ios::in);
    assert(!ifs.fail()); // makes sure file open
    string line; // line holder variable
    string item; // temp variable for current line read/parsed
    vector<int> items; // list of all itmes read
    while (getline(ifs, line)) { // goes through all line of file
        stringstream ss(line); // makes line in string stream
        ss >> item; // extract the item from the current line
        items.push_back(stoi(item)); // stores item for returning
    }
    ifs.close(); // closes the file
    return items; // returns list of items
}


vector<string> filter_missing_ents(vector<string> all_entities, vector<string> missing_entities) {
    unsigned i = 0;
    while (i<all_entities.size()) {
        auto position = find(missing_entities.begin(),missing_entities.end(),all_entities[i]);
        if (position == missing_entities.end()) {
            i++;
        } else {
            all_entities.erase(all_entities.begin()+i);
        }
    }
    return all_entities;
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

void train_initial(int efreq, const string& model_path, const int neg_ratio,
                   const int num_thread, const int num_epoch, const int ne,
                   const int nr, const vector<triplet>& sros,
                   Evaluator& etr, Evaluator& eva,
                   Model* model, SROBucket* bucket,
                   const unordered_map<string,int>& ent_map,
                   const unordered_map<string, int>& rel_map,
                   SROBucket* skip_bucket){
    // thread-specific negative samplers
    vector<NegativeSampler> neg_samplers;
    for (int tid = 0; tid < num_thread; tid++) {
        // creates a bunch of random entity/relation generators
        neg_samplers.push_back(NegativeSampler(ne, nr, rand() ^ tid,
                                               ent_map, rel_map, sros) );
    }

    int N = sros.size(); // N is number of examples in training set
    vector<int> pi = range(N); // pi is the range of numbers in N

    double best_mrr = 0;
    omp_set_num_threads(num_thread); // tells omp lib num of threads to use
    for (int epoch = 0; epoch < num_epoch; epoch++) {
        // goes through the full number of epochs for training
        if (epoch % efreq == 0) {
            // evaluation
            auto info_tr = etr.evaluate(model, bucket, skip_bucket, 2048);
            auto info_va = eva.evaluate(model, bucket, skip_bucket, 2048);
            // save the best model to disk
            double curr_mrr = condense_mrr(info_va);
            if (curr_mrr > best_mrr) {
                best_mrr = curr_mrr;
                if ( !model_path.empty() )
                    model->save(model_path);
            }
            //eval_print("VALID EVALUATION",info_va);
            eval_log("VALID EVALUATION",info_va);
        }

        // shuffles all the numbers corresponding to each example
        shuffle(pi.begin(), pi.end(), GLOBAL_GENERATOR);

        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            // goes through each example
            triplet sro = sros[pi[i]];
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

                model->train(s, r, oo, false);
                model->train(ss, r, o, false);
                model->train(s, rr, o, false);
            }
        }
    }
}

void train_additional(const int neg_ratio, const int num_thread,
                      const vector<triplet>& sros,
                      Evaluator& given_evaluator, Evaluator& missing_evaluator,
                      Evaluator& combined_evaluator,
                      Model* model, SROBucket* truth_bucket,
                      const unordered_map<string,int>& ent_map,
                      const unordered_map<string, int>& rel_map,
                      SROBucket* skip_bucket, SROBucket* combined_skip_bucket,
                      string log_file_name,
                      string model_path, vector<int> missing_ent_ids,
                      int num_miss, int miss_set){

    // thread-specific negative samplers
    vector<NegativeSampler> neg_samplers;
    for (int tid = 0; tid < num_thread; tid++) {
        // creates a bunch of random entity/relation generators
        neg_samplers.push_back(NegativeSampler(ent_map.size(), rel_map.size(), rand() ^ tid,
                                               ent_map, rel_map, sros) );
    }

    int N = sros.size(); // N is number of examples in training set
    vector<int> pi = range(N); // pi is the range of numbers in N

    omp_set_num_threads(num_thread); // tells omp lib num of threads to use

    int num_epoch = 100;
    double m_conv_thresh = 0.01;
    double g_conv_thresh = 0.01;

    // get batch learning average performance for missing/given data
    double best_given = get_batch_learn_performance(given_evaluator,truth_bucket,skip_bucket,model_path,-1,missing_ent_ids,num_miss,miss_set,"g");
    double best_missing = get_batch_learn_performance(missing_evaluator,truth_bucket,skip_bucket,model_path,-1,missing_ent_ids,num_miss,miss_set,"m");
    //double best_combined = get_batch_learn_performance(combined_evaluator,truth_bucket,combined_skip_bucket,model_path,-1,missing_ent_ids,num_miss,miss_set,"c");
    // TODO add the combined threshold later?
    // begin logging the 'continual learning' performance
    Logger* cl_log = new Logger();
    for (int epoch = 0; epoch < num_epoch; epoch++) {
        // evaluation
        auto info_m = missing_evaluator.evaluate(model, truth_bucket, skip_bucket, -1);
        auto info_g = given_evaluator.evaluate(model, truth_bucket, skip_bucket, -1);
        auto info_c = combined_evaluator.evaluate(model, truth_bucket, combined_skip_bucket, -1);

        double mrr_m = condense_mrr(info_m);
        double mrr_g = condense_mrr(info_g);
        double mrr_c = condense_mrr(info_c);

        vector<double> new_sample{(double) epoch,mrr_g,mrr_m,mrr_c};
        cl_log->new_sample(new_sample);
        if ( ((best_missing-mrr_m) < m_conv_thresh) && ((best_given-mrr_g) < g_conv_thresh) ) {
            break;
        }

        // shuffles all the numbers corresponding to each example
        shuffle(pi.begin(), pi.end(), GLOBAL_GENERATOR);

        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            // goes through each example
            triplet sro = sros[pi[i]];
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

                model->train(s, r, oo, false);
                model->train(ss, r, o, false);
                model->train(s, rr, o, false);
            }
        }
    }
    cl_log->flush_log(log_file_name);
    delete cl_log;
}

int main(int argc, char **argv) {
    // default options
    string  ddir        =  "./datasets/";
    string  dataset     =  "sd_thor"; // training set
    string  experiment  =  "tg_all_0"; // training set
    string  ins_method  =  "w2v";
    int     embed_dim   =  100; // dimensionality of the embedding
    double  eta         =  0.1; // related to learning rate
    double  gamma       =  1e-3; // related to gradient
    int     neg_ratio   =  9; // number of negative examples to see
    int     num_epoch   =  500; // number of epochs
    int     num_thread  =  1; // number of threads
    int     eval_freq   =  10; // how often to evaluate while training
    string  model_path; // path to output- (train) or input- (test) model
    int     num_scalar  = embed_dim / 2.0;
    int     num_missing = 1;
    int     phase       =  0; // whether testing or training
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
    if ((i = ArgPos((char *)"-ins_method", argc, argv)) > 0)  ins_method  =  string(argv[i+1]);
    if ((i = ArgPos((char *)"-num_scalar", argc, argv)) > 0)  num_scalar  =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-num_missing", argc, argv)) > 0) num_missing =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-phase", argc, argv)) > 0)       phase       =  atoi(argv[i+1]);
    num_scalar  = embed_dim / 2.0;
    // lists all the ANALOGY arguments
    printf("ins_method  =  %s\n", ins_method.c_str());
    printf("dataset     =  %s\n", dataset.c_str());
    printf("experiment  =  %s\n", experiment.c_str());
    printf("embed_dim   =  %d\n", embed_dim);
    printf("eta         =  %e\n", eta);
    printf("gamma       =  %e\n", gamma);
    printf("neg_ratio   =  %d\n", neg_ratio);
    printf("num_epoch   =  %d\n", num_epoch);
    printf("num_thread  =  %d\n", num_thread);
    printf("eval_freq   =  %d\n", eval_freq);
    model_path = dataset + "_" + experiment;
    printf("model_path  =  %s\n", model_path.c_str());
    printf("num_scalar  =  %d\n", num_scalar);
    printf("num_missing  = %d\n", num_missing);
    printf("phase       =  %d\n", phase);
    // gets entities & relations as strings
    vector<string> ents = read_first_column(ddir+dataset+"_entities.csv");
    vector<string> rels = read_first_column(ddir+dataset+"_relations.csv");
    // creates mapping between entities/relations and unique ID
    unordered_map<string, int> ent_map = create_id_mapping(ents);
    unordered_map<string, int> rel_map = create_id_mapping(rels);
    int ne = ent_map.size();
    int nr = rel_map.size();
    // initializes the ANALOGY model pointer
    Model *model = NULL;

    vector<triplet> sros_tr_given;
    vector<triplet> sros_va_given;
    vector<triplet> sros_te_given;
    vector<triplet> sros_tr_missing;
    vector<triplet> sros_va_missing;
    vector<triplet> sros_te_missing;
    vector<triplet> sros_to_train;
    vector<triplet> sros_al;
    vector<triplet> sros_skip;
    vector<triplet> combined_sros_skip;
    vector<triplet> missing;
    vector<triplet> combined;

    // for immediate evaulations
    vector<vector<vector<double>>> immediate_given_avg;
    vector<vector<vector<double>>> immediate_missing_avg;
    vector<vector<vector<double>>> immediate_combined_avg;
    vector<vector<vector<double>>> immediate_given_totals;
    vector<vector<vector<double>>> immediate_missing_totals;
    vector<vector<vector<double>>> immediate_combined_totals;
    initialize_info(immediate_given_avg,nr);
    initialize_info(immediate_missing_avg,nr);
    initialize_info(immediate_combined_avg,nr);
    initialize_info(immediate_given_totals,nr);
    initialize_info(immediate_missing_totals,nr);
    initialize_info(immediate_combined_totals,nr);
    // for final evaluations
    vector<vector<vector<double>>> final_given_avg;
    vector<vector<vector<double>>> final_missing_avg;
    vector<vector<vector<double>>> final_combined_avg;
    vector<vector<vector<double>>> final_given_totals;
    vector<vector<vector<double>>> final_missing_totals;
    vector<vector<vector<double>>> final_combined_totals;
    initialize_info(final_given_avg,nr);
    initialize_info(final_missing_avg,nr);
    initialize_info(final_combined_avg,nr);
    initialize_info(final_given_totals,nr);
    initialize_info(final_missing_totals,nr);
    initialize_info(final_combined_totals,nr);

    // generate missing entity sets
    int num_missing_sets = 30;
    vector<string> missing_ent_fps = get_missing_ent_files(ne,num_missing,num_missing_sets,model_path);

    for (int missing_set_idx=0;missing_set_idx<num_missing_sets;missing_set_idx++) {
        // selects the entities that will be missing
        string missing_ent_fp = missing_ent_fps[missing_set_idx];
        vector<int> missing_ent_ids = get_missing_ent_ids(missing_ent_fp);
        vector<string> missing_ents;
        for (unsigned missing_ent_idx=0;missing_ent_idx<missing_ent_ids.size();missing_ent_idx++)
            missing_ents.push_back(ents[missing_ent_ids[missing_ent_idx]]);
        cout << "Working on missing entity(s): ";
        for (unsigned missing_ent_idx=0;missing_ent_idx<missing_ent_ids.size();missing_ent_idx++)
            cout << missing_ents[missing_ent_idx] << "  ";
        cout << endl;

        string model_path_e_prime = model_path+"_multE_"+to_string(num_missing)+"_"+to_string(missing_set_idx);
        // removes from ents
        vector<string> initial_ents = filter_missing_ents(ents,missing_ents);

        // makes new entity map to remove missing
        unordered_map<string,int> initial_ent_map = create_id_mapping(initial_ents);
        int initial_ne = initial_ent_map.size();
        unordered_map<string,int> missing_ent_map = create_id_mapping(missing_ents,initial_ne);
        vector<int> new_missing_ent_ids;
        for (unsigned missing_ent_idx=0;missing_ent_idx<missing_ent_ids.size();missing_ent_idx++)
            new_missing_ent_ids.push_back(missing_ent_map[missing_ents[missing_ent_idx]]);

        // create analogy model
        model = new Analogy(initial_ne,nr,embed_dim,num_scalar,eta,gamma);
        assert(model != NULL);

        // create triplets for given and missing train, valid, and test sets
        string triples_fp_root = ddir+dataset+"_"+experiment;
        create_sros(sros_tr_given,sros_tr_missing,triples_fp_root+"_train.csv",initial_ent_map,rel_map,missing_ent_map);
        create_sros(sros_va_given,sros_va_missing,triples_fp_root+"_valid.csv",initial_ent_map,rel_map,missing_ent_map);
        create_sros(sros_te_given,sros_te_missing,triples_fp_root+"_test.csv",initial_ent_map,rel_map,missing_ent_map);

        // initial training over only !E' data if model does not exist
        if (!model_exists(model_path_e_prime)) {
            // creates a buckets of !E' s,r,o triplets for training
            sros_to_train.clear();
            sros_skip.clear();
            sros_to_train.insert(sros_to_train.end(), sros_tr_given.begin(), sros_tr_given.end());
            sros_to_train.insert(sros_to_train.end(), sros_va_given.begin(), sros_va_given.end());
            sros_to_train.insert(sros_to_train.end(), sros_te_given.begin(), sros_te_given.end());
            sros_skip.insert(sros_skip.end(), sros_tr_given.begin(), sros_tr_given.end());
            SROBucket sro_bucket_al_given(sros_to_train);
            SROBucket* sro_bucket_given = &sro_bucket_al_given;
            SROBucket sro_bucket_skip(sros_skip);
            SROBucket *skip_bucket = &sro_bucket_skip;
            // evaluator for validation data
            Evaluator evaluator_va(initial_ne, nr, sros_va_given);
            // evaluator for training data
            Evaluator evaluator_tr(initial_ne, nr, sros_tr_given);
            // train the !E' embedding
            cout << "Training Initial Model..." << endl;
            train_initial(eval_freq,model_path_e_prime,neg_ratio,num_thread,num_epoch,
                          initial_ne,nr,sros_tr_given,evaluator_tr,evaluator_va,
                          model,sro_bucket_given,initial_ent_map,rel_map,
                          skip_bucket);
        }

        // put E' back in entity set
        for (unsigned missing_ent_idx=0;missing_ent_idx<missing_ents.size();missing_ent_idx++) {
            string missing_ent = missing_ents[missing_ent_idx];
            initial_ents.push_back(missing_ent);
            initial_ent_map[missing_ent] = missing_ent_map[missing_ent];
        }
        int new_ne = initial_ent_map.size();

        // prepare buckets for additional train/evaluations
        sros_al.clear();
        sros_skip.clear();
        combined_sros_skip.clear();
        missing.clear();
        sros_al.insert(sros_al.end(), sros_tr_given.begin(), sros_tr_given.end());
        sros_al.insert(sros_al.end(), sros_va_given.begin(), sros_va_given.end());
        sros_al.insert(sros_al.end(), sros_te_given.begin(), sros_te_given.end());
        sros_al.insert(sros_al.end(), sros_tr_missing.begin(), sros_tr_missing.end());
        sros_al.insert(sros_al.end(), sros_va_missing.begin(), sros_va_missing.end());
        sros_al.insert(sros_al.end(), sros_te_missing.begin(), sros_te_missing.end());
        missing.insert(missing.end(), sros_tr_missing.begin(), sros_tr_missing.end());
        missing.insert(missing.end(), sros_va_missing.begin(), sros_va_missing.end());
        missing.insert(missing.end(), sros_te_missing.begin(), sros_te_missing.end());
        sros_skip.insert(sros_skip.end(), sros_tr_given.begin(), sros_tr_given.end());
        sros_skip.insert(sros_skip.end(), sros_va_given.begin(), sros_va_given.end());
        combined_sros_skip.insert(combined_sros_skip.end(), sros_tr_given.begin(), sros_tr_given.end());
        combined_sros_skip.insert(combined_sros_skip.end(), sros_va_given.begin(), sros_va_given.end());
        combined_sros_skip.insert(combined_sros_skip.end(), sros_tr_missing.begin(), sros_tr_missing.end());
        combined_sros_skip.insert(combined_sros_skip.end(), sros_va_missing.begin(), sros_va_missing.end());
        SROBucket sro_bucket_al(sros_al);
        SROBucket* sro_bucket_all = &sro_bucket_al;
        SROBucket* skip_bucket = new SROBucket(sros_skip);
        SROBucket* combined_skip_bucket = new SROBucket(combined_sros_skip);

        // load model !E'
        delete model;
        double eta2 = 2e-3;
        double gamma2 = 1e-1;
        model = new Analogy(initial_ne,nr,embed_dim,num_scalar,eta2,gamma2);
        assert(model != NULL);
        model->load(model_path_e_prime);

        // insert E' via a method
        if (ins_method == "xavier") {
            for (int j=0;j<num_missing;j++) {
                model->E.push_back(uniform_array(embed_dim,-0.6,0.6));
                model->E_g.push_back(const_array(embed_dim,1e-6));
            }
        } else if (ins_method == "random") {
            for (int j=0;j<num_missing;j++) {
                model->E.push_back(uniform_array_from_E(embed_dim,model->E));
                model->E_g.push_back(const_array(embed_dim,1e-6));
            }
        } else if (ins_method == "similarity") {
            SimilarityInsert* sim_guesser = new SimilarityInsert(model->E,new_ne,"./datasets/w2v_"+dataset+"_entity_embeddings.csv");
            vector<vector<double>> initial_embeddings = sim_guesser->insert_entity(missing_ent_ids);
            for (unsigned j=0;j<initial_embeddings.size();j++) {
                model->E.push_back(initial_embeddings[j]);
                model->E_g.push_back(const_array(embed_dim,1e-6));
            }
            delete sim_guesser;
        } else if (ins_method == "similarity2") {
            Similarity2Insert* sim_guesser = new Similarity2Insert(model->E,new_ne,"./datasets/w2v_"+dataset+"_entity_embeddings.csv");
            vector<vector<double>> initial_embeddings = sim_guesser->insert_entity(missing_ent_ids);
            for (unsigned j=0;j<initial_embeddings.size();j++) {
                model->E.push_back(initial_embeddings[j]);
                model->E_g.push_back(const_array(embed_dim,1e-6));
            }
            delete sim_guesser;
        } else if (ins_method == "relational_node") {
            // loads all UNIQUE triples, randomly keeps x% for insert
            auto insert_triples = select_insert_triples(missing,new_missing_ent_ids);
            RelationalNodeInsert* rel_guesser = new RelationalNodeInsert(model->E,model->R,insert_triples,embed_dim,num_scalar);
            vector<vector<double>> initial_embeddings = rel_guesser->insert_entity(new_missing_ent_ids);
            for (unsigned j=0;j<initial_embeddings.size();j++) {
                model->E.push_back(initial_embeddings[j]);
                model->E_g.push_back(const_array(embed_dim,1e-6));
            }
            delete rel_guesser;
        } else if (ins_method == "hybrid") {
            // loads all UNIQUE triples, randomly keeps x% for insert
            auto insert_triples = select_insert_triples(missing,new_missing_ent_ids);
            HybridInsert* hybrid_guesser = new HybridInsert(model->E,model->R,insert_triples,embed_dim,num_scalar,new_ne,"./datasets/w2v_"+dataset+"_entity_embeddings.csv");
            vector<vector<double>> initial_embeddings = hybrid_guesser->insert_entity(missing_ent_ids,new_missing_ent_ids);
            for (unsigned j=0;j<initial_embeddings.size();j++) {
                model->E.push_back(initial_embeddings[j]);
                model->E_g.push_back(const_array(embed_dim,1e-6));
            }
            delete hybrid_guesser;
        }

        if (phase == 2) {
            sros_al.clear();
            sros_al.insert(sros_al.end(), sros_tr_given.begin(), sros_tr_given.end());
            sros_al.insert(sros_al.end(), sros_va_given.begin(), sros_va_given.end());
            SROBucket sro_bucket_ali(sros_al);
            SROBucket *sro_bucket_alli = &sro_bucket_ali;
            InteractiveTerminal interface(new_ne, nr);
            do {
                interface.query(model,sro_bucket_alli,ent_map,rel_map);
            } while (interface.continue_interaction());
            return 0;
        }

        // immediate evaluation on given/missing
        Evaluator evaluator_given(new_ne, nr, sros_te_given);
        Evaluator evaluator_missing(new_ne, nr, missing);

        // TODO added as new output
        combined.clear();
        combined.insert(combined.end(), sros_te_missing.begin(), sros_te_missing.end());
        combined.insert(combined.end(), sros_te_given.begin(), sros_te_given.end());
        Evaluator evaluator_combined(new_ne, nr, combined);

        auto info_given = evaluator_given.evaluate(model, sro_bucket_all, skip_bucket, -1);
        auto info_missing = evaluator_missing.evaluate(model, sro_bucket_all, skip_bucket, -1);
        auto info_combined = evaluator_combined.evaluate(model, sro_bucket_all, combined_skip_bucket, -1);

        sum_info(immediate_given_avg,info_given,immediate_given_totals);
        sum_info(immediate_missing_avg,info_missing,immediate_missing_totals);
        sum_info(immediate_combined_avg,info_combined,immediate_combined_totals);

        // prepare 'additional training' triples (both E' and !E')
        sros_to_train.clear();
        sros_to_train.insert(sros_to_train.end(), sros_tr_given.begin(), sros_tr_given.end());
        sros_to_train.insert(sros_to_train.end(), sros_tr_missing.begin(), sros_tr_missing.end());
        //Evaluator evaluator_va(initial_ne, nr, sros_va_given);

        // construct the sros vector to train on!
        cout << "Additional training in progress..." << endl;
        train_additional(neg_ratio, num_thread, sros_to_train,
                         evaluator_given, evaluator_missing, evaluator_combined,
                         model, sro_bucket_all, initial_ent_map, rel_map,
                         skip_bucket, combined_skip_bucket,
                         model_path_e_prime + "_" + ins_method,
                         model_path, missing_ent_ids,
                         num_missing,missing_set_idx);

        info_given = evaluator_given.evaluate(model, sro_bucket_all, skip_bucket, -1);
        info_missing = evaluator_missing.evaluate(model, sro_bucket_all, skip_bucket, -1);
        info_combined = evaluator_combined.evaluate(model, sro_bucket_all, combined_skip_bucket, -1);

        sum_info(final_given_avg,info_given,final_given_totals);
        sum_info(final_missing_avg,info_missing,final_missing_totals);
        sum_info(final_combined_avg,info_combined,final_combined_totals);

        delete skip_bucket;
        delete combined_skip_bucket;
        delete model;
    }

    average_totals(immediate_given_avg,immediate_given_totals);
    average_totals(immediate_missing_avg,immediate_missing_totals);
    average_totals(immediate_combined_avg,immediate_combined_totals);
    average_totals(final_given_avg,final_given_totals);
    average_totals(final_missing_avg,final_missing_totals);
    average_totals(final_combined_avg,final_combined_totals);

    eval_print("IMMEDIATE ONLY !E' EVALUATION",immediate_given_avg);
    eval_print("IMMEDIATE ONLY E' EVALUATION",immediate_missing_avg);
    eval_print("IMMEDIATE COMBINED EVALUATION",immediate_combined_avg);
    eval_print("FINAL E' & !E' EVALUATION",final_given_avg);
    eval_print("FINAL ONLY E' EVALUATION",final_missing_avg);
    eval_print("FINAL COMBINED EVALUATION",final_combined_avg);
    // save outputs
    eval_csv("./robocse_logging/immediate_given_avg_"+to_string(num_missing)+"_"+ins_method+".csv",immediate_given_avg);
    eval_csv("./robocse_logging/immediate_missing_avg_"+to_string(num_missing)+"_"+ins_method+".csv",immediate_missing_avg);
    eval_csv("./robocse_logging/immediate_combined_avg_"+to_string(num_missing)+"_"+ins_method+".csv",immediate_combined_avg);

    eval_csv("./robocse_logging/final_given_avg_"+to_string(num_missing)+"_"+ins_method+".csv",final_given_avg);
    eval_csv("./robocse_logging/final_missing_avg_"+to_string(num_missing)+"_"+ins_method+".csv",final_missing_avg);
    eval_csv("./robocse_logging/final_combined_avg_"+to_string(num_missing)+"_"+ins_method+".csv",final_combined_avg);

    return 0;
}

