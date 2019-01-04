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

void create_sros(int missing_id,
                 vector<triplet>& given_sros, vector<triplet>& missing_sros,
                 const string& fname,
                 const unordered_map<string, int>& ent_map,
                 const unordered_map<string, int>& rel_map) {
    // creates a dataset of s,r,o triplets
    ifstream ifs(fname, ios::in); // file of triplets
    // make sure dataset file is open
    assert(!ifs.fail());

    string line; // triplet line variable
    string s, r, o; // subj, obj, rel holders
    int s_id,o_id; // subj/obj id holders
    given_sros.clear();
    missing_sros.clear();

    getline(ifs, line); // skip first line
    while (getline(ifs,line)) { // go through all lines in dataset
        stringstream ss(line);
        getline(ss,s,',');
        getline(ss,r,',');
        getline(ss,o,',');
        // check for subject and object in entity map, o.w. skips triplet
        try {
            s_id = ent_map.at(s);
        } catch (const out_of_range& e) {
            s_id = missing_id;
        }
        try {
            o_id = ent_map.at(o);
        } catch (const out_of_range& e) {
            o_id = missing_id;
        }
        // add triplet to list while mapping names to unique IDs
        if ( (s_id == missing_id) || (o_id == missing_id) ){
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
                __counts[__sro] = 0;
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

class Evaluator {
    int ne;
    int nr;
    const vector<triplet>& sros;

public:
    Evaluator(int ne, int nr, const vector<triplet>& sros) :
        // sets the number of relations and entities, also gives the set of 
        // s,r,o used to either test, train, or validate and the complete s,r,o
        // bucket        
        ne(ne), nr(nr), sros(sros) {}

    std::vector<std::vector<std::vector<double>>> evaluate(const Model *model, const SROBucket *bucket, int truncate) {
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
    }

private:

    tuple<double, double, double> rank(const SROBucket *model, const triplet& sro) {
        int rank_s = 1;
        int rank_r = 1;
        int rank_o = 1;

        int s = get<0>(sro);
        int r = get<1>(sro);
        int o = get<2>(sro);

        double base_score = double(model->score(s, r, o));

        for (int ss = 0; ss < ne; ss++)
            if (model->score(ss, r, o) > base_score) rank_s++;

        for (int rr = 0; rr < nr; rr++)
            if (model->score(s, rr, o) > base_score) rank_r++;

        for (int oo = 0; oo < ne; oo++)
            if (model->score(s, r, oo) > base_score) rank_o++;

        return make_tuple(rank_s, rank_r, rank_o);
    }

    tuple<double, double, double> rank(const Model *model, const triplet& sro) {
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

        for (int ss = 0; ss < ne; ss++)
            if (model->score(ss, r, o) > base_score) rank_s++;

        for (int rr = 0; rr < nr; rr++)
            if (model->score(s, rr, o) > base_score) rank_r++;

        for (int oo = 0; oo < ne; oo++)
            if (model->score(s, r, oo) > base_score) rank_o++;

        return make_tuple(rank_s, rank_r, rank_o);
    }
};

void eval_log(const char* prefix, const vector<vector<vector<double>>>& info) {
    FILE * pFile;
    pFile = fopen("logfile.txt","a");
    fprintf(pFile,"------------------%s---------------------\n", prefix);
    fprintf(pFile," Query | Relation |    MRR | Hits10 |  Hits5 |  Hits1 \n");
    fprintf(pFile,"  Sro  | atLoc    | %6.2f | %6.2f | %6.2f | %6.2f \n", 100.0*info[0][0][0],100.0*info[0][0][1],100.0*info[0][0][2],100.0*info[0][0][3]);
    fprintf(pFile,"  Sro  | hasMat   | %6.2f | %6.2f | %6.2f | %6.2f \n", 100.0*info[0][1][0],100.0*info[0][1][1],100.0*info[0][1][2],100.0*info[0][1][3]);
    fprintf(pFile,"  Sro  | hasAff   | %6.2f | %6.2f | %6.2f | %6.2f \n", 100.0*info[0][2][0],100.0*info[0][2][1],100.0*info[0][2][2],100.0*info[0][2][3]);
    fprintf(pFile,"  sRo  |          | %6.2f | %6.2f | %6.2f | %6.2f \n", 100.0*info[1][0][0],100.0*info[1][0][1],100.0*info[1][0][2],100.0*info[1][0][3]);
    fprintf(pFile,"  srO  | atLoc    | %6.2f | %6.2f | %6.2f | %6.2f \n", 100.0*info[2][0][0],100.0*info[2][0][1],100.0*info[2][0][2],100.0*info[2][0][3]);
    fprintf(pFile,"  srO  | hasMat   | %6.2f | %6.2f | %6.2f | %6.2f \n", 100.0*info[2][1][0],100.0*info[2][1][1],100.0*info[2][1][2],100.0*info[2][1][3]);
    fprintf(pFile,"  srO  | hasAff   | %6.2f | %6.2f | %6.2f | %6.2f \n", 100.0*info[2][2][0],100.0*info[2][2][1],100.0*info[2][2][2],100.0*info[2][2][3]);
    fclose(pFile);
}

void eval_print(const char* prefix, const vector<vector<vector<double>>>& info) {
    printf("------------------%s---------------------\n", prefix);
    printf(" Query | Relation |    MRR | Hits10 |  Hits5 |  Hits1 \n");
    printf("  Sro  | atLoc    | %6.2f | %6.2f | %6.2f | %6.2f \n", 100.0*info[0][0][0],100.0*info[0][0][1],100.0*info[0][0][2],100.0*info[0][0][3]);
    printf("  Sro  | hasMat   | %6.2f | %6.2f | %6.2f | %6.2f \n", 100.0*info[0][1][0],100.0*info[0][1][1],100.0*info[0][1][2],100.0*info[0][1][3]);
    printf("  Sro  | hasAff   | %6.2f | %6.2f | %6.2f | %6.2f \n", 100.0*info[0][2][0],100.0*info[0][2][1],100.0*info[0][2][2],100.0*info[0][2][3]);
    printf("  sRo  |          | %6.2f | %6.2f | %6.2f | %6.2f \n", 100.0*info[1][0][0],100.0*info[1][0][1],100.0*info[1][0][2],100.0*info[1][0][3]);
    printf("  srO  | atLoc    | %6.2f | %6.2f | %6.2f | %6.2f \n", 100.0*info[2][0][0],100.0*info[2][0][1],100.0*info[2][0][2],100.0*info[2][0][3]);
    printf("  srO  | hasMat   | %6.2f | %6.2f | %6.2f | %6.2f \n", 100.0*info[2][1][0],100.0*info[2][1][1],100.0*info[2][1][2],100.0*info[2][1][3]);
    printf("  srO  | hasAff   | %6.2f | %6.2f | %6.2f | %6.2f \n", 100.0*info[2][2][0],100.0*info[2][2][1],100.0*info[2][2][2],100.0*info[2][2][3]);
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
    for(unsigned i = 0; i < 3; i++){
        for(unsigned j = 0; j < metric_total[0].size(); j++){
            for(unsigned k = 0; k < 4; k++){
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

void train_model(int efreq, const string& model_path, const int neg_ratio,
                 const int num_thread, const int num_epoch, const int ne,
                 const int nr, const vector<triplet>& sros,
                 Evaluator& etr, Evaluator& eva,
                 Model* model, SROBucket* bucket){
    // thread-specific negative samplers
    vector<NegativeSampler> neg_samplers;
    for (int tid = 0; tid < num_thread; tid++) {
        // creates a bunch of random entity/relation generators
        neg_samplers.push_back( NegativeSampler(ne, nr, rand() ^ tid) );
    }

    int N = sros.size(); // N is number of examples in training set
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
        if (epoch % efreq == 0) {
            // evaluation
            start_e = omp_get_wtime();
            auto info_tr = etr.evaluate(model, bucket, 2048);
            auto info_va = eva.evaluate(model, bucket, 2048);
            elapse_ev = omp_get_wtime() - start_e;
            printf("Elapse EV    %f\n", elapse_ev);
            // save the best model to disk
            double curr_mrr = condense_mrr(info_va);
            if (curr_mrr > best_mrr) {
                best_mrr = curr_mrr;
                if ( !model_path.empty() )
                    model->save(model_path+".model");
                    cout << "Model saved." << endl;
            }
            //eval_print("TRAIN EVALUATION",info_tr);
            eval_print("VALID EVALUATION",info_va);
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
                int oo = neg_samplers[tid].random_entity();
                int ss = neg_samplers[tid].random_entity();
                int rr = neg_samplers[tid].random_relation();

                model->train(s, r, oo, false);
                model->train(ss, r, o, false);
                model->train(s, rr, o, false);
            }
        }
    }
    elapse_tr = omp_get_wtime() - start_t;
    printf("Elapse TR    %f\n", elapse_tr);
}

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
    int     num_scalar  = embed_dim / 2.0;
    int     train_size  = 0;
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
    if ((i = ArgPos((char *)"-num_scalar", argc, argv)) > 0)  num_scalar  =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-train_size", argc, argv)) > 0)  train_size  =  atoi(argv[i+1]);
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
    vector<triplet> sros_al;

    for (int entity_id=0; entity_id<ne; entity_id++) {
        // selects the entity that will be missing
        string missing_entity = ents[entity_id];
        int missing_entity_id = ent_map.at(missing_entity);
        // removes from ents
        vector<string> initial_ents(ents);
        initial_ents.erase(initial_ents.begin()+missing_entity_id);
        // makes new entity map to remove missing
        unordered_map<string,int> initial_ent_map = create_id_mapping(initial_ents);
        int initial_ne = initial_ent_map.size();
        // clears the previous model
        model = new Analogy(initial_ne,nr,embed_dim,num_scalar,eta,gamma);
        assert(model != NULL);
        // create triplets for given and missing train, valid, and test sets
        string triples_fp_root = ddir+dataset+"_"+experiment;
        create_sros(missing_entity_id,sros_tr_given,sros_tr_missing,triples_fp_root+"_train.csv",initial_ent_map,rel_map);
        create_sros(missing_entity_id,sros_va_given,sros_va_missing,triples_fp_root+"_valid.csv",initial_ent_map,rel_map);
        create_sros(missing_entity_id,sros_te_given,sros_te_missing,triples_fp_root+"_test.csv",initial_ent_map,rel_map);
        // initial training over only GIVEN data, makes ground truth
        sros_al.clear();
        sros_al.insert(sros_al.end(), sros_tr_given.begin(), sros_tr_given.end());
        sros_al.insert(sros_al.end(), sros_va_given.begin(), sros_va_given.end());
        sros_al.insert(sros_al.end(), sros_te_given.begin(), sros_te_given.end());
        // creates a 'bucket' object of all s,r,o triplets, used later
        SROBucket sro_bucket_al(sros_al);
        SROBucket* sro_bucket = &sro_bucket_al;
        // evaluator for validation data
        Evaluator evaluator_va(initial_ne, nr, sros_va_given);
        // evaluator for training data
        Evaluator evaluator_tr(initial_ne, nr, sros_tr_given);
        // train the embedding without the entity
        train_model(eval_freq,model_path,neg_ratio,num_thread,num_epoch,initial_ne,nr,
                    sros_tr_given,evaluator_tr,evaluator_va,model,sro_bucket);
        cout << "hello" << endl;
        // reinsert the missing entity, and evaluate immediately on given/missing valid
        // retrain over all (given/missing) data
    }

    return 0;
}

