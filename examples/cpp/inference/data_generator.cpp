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
#include "data_generator.h"
using namespace std;

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
