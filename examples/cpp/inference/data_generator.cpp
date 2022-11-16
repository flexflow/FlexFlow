//
//  main.cpp
//  dataloader
//
//  Created by User on 11/15/22.
//

#include <iostream>
#include <ctime>
#include <random>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <math.h>
using namespace std;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds milliseconds;

class Generator {
    public:
        size_t num_requests; // total number of requests
        size_t tensor_size; // dimension of one request tensor
        bool poisson; // false implied uniform distribution
        double lambda; // mean #num of arrivals per sec
        
        Generator(size_t req, size_t tensor, bool poi, double lamb) {
            num_requests = req;
            tensor_size = tensor;
            poisson = poi;
            lambda = lamb;
            compute_distribution();
            arrivals_ptr = arrivals.begin();
            timer_started = false;
        }
    
        vector<vector<double>> get_requests(void); // function to retrieve requests
    
    private:
        bool timer_started; // tracks if start time has been initiated
        Clock::time_point start_time; // time when get_requests() is called for the first time
        vector<double> arrivals; // arrival times (ms) generated based on distribution
        vector<double>::iterator arrivals_ptr; // next request to output
    
        void compute_distribution( void ); // populate arrivals
        vector<double> get_random_tensor(void); // generate a random tensor
};

void Generator::compute_distribution( void ) {
    // set up uniform number generator [0,1)
    random_device rnd;
    mt19937 gen(rnd());
    uniform_real_distribution<double> dist {0, 1.0};
    double cur_arrival = 0; // assume first request comes in at time 0
    
    for (size_t i = 0; i < num_requests; i++) {
        arrivals.push_back(cur_arrival);
        cout << "arrival time " << i << ": +" << cur_arrival << "ms \n";
        
        if (poisson) {
            double u = dist(gen);
            double interval = -(1/lambda) * log(1-u) * 1000;
            cur_arrival += interval;
        } else {
            cur_arrival += (1000/lambda);
        }
    }
    return;
};

vector<vector<double>> Generator::get_requests(void) {
    Clock::time_point cur_time = Clock::now();
    vector<vector<double>> requests;
    if (!timer_started){
        // simply return one request and start timer for the first call
        start_time = Clock::now();
        timer_started = true;
        arrivals_ptr++;
        requests.push_back(get_random_tensor());
        return requests;
    }
    
    // output requests till we reach current timestamp
    milliseconds ms_from_start = chrono::duration_cast<milliseconds>(cur_time - start_time);
    while (arrivals_ptr < arrivals.end() && ms_from_start.count() >= *arrivals_ptr){
        cout << "output request at arrival time +" << *arrivals_ptr << "\n";
        requests.push_back(get_random_tensor());
        arrivals_ptr++;
    }
    return requests;
};

vector<double> Generator::get_random_tensor(void) {
    random_device rnd_device;
    mt19937 mersenne_engine {rnd_device()};
    uniform_real_distribution<double> dist {0, 1.0}; // state distribution
    
    auto gen = [&dist, &mersenne_engine](){
                   return dist(mersenne_engine);
               };

    vector<double> vec(tensor_size);
    generate(begin(vec), end(vec), gen);
    return vec;
};

// for debugging
void print_requests(vector<vector<double>> req) {
    cout << "printing requests\n";
    for (vector<double> v: req){
        for (double e: v) {
            cout << e << ",";
        }
        cout << "\n";
    }
    cout << "\n";
};

int main(int argc, const char * argv[]) {
    // insert code here...
    cout << "Hello, World!\n";
    Generator data_generator(10, 4, true, 1);
    
    vector<vector<double>> req0 = data_generator.get_requests();
    print_requests(req0);
    
    this_thread::sleep_for(milliseconds(1200));
    vector<vector<double>> req1200 = data_generator.get_requests();
    print_requests(req1200);
    
    this_thread::sleep_for(milliseconds(10));
    vector<vector<double>> req1210 = data_generator.get_requests();
    print_requests(req1210);
    
    this_thread::sleep_for(milliseconds(4000));
    vector<vector<double>> req5210 = data_generator.get_requests();
    print_requests(req5210);
    
    
    return 0;
}
