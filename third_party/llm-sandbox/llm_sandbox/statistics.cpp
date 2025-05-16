#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cstdlib>
#include <algorithm>

// Calculate the area under the curve using trapezoidal rule
double calculate_efficiency_integral(const std::vector<long long>& times, const std::vector<long long>& memories) {
    if (times.size() != memories.size()) {
        return 0.0;
    }
    
    if (times.size() < 2) {
        return 0.0;
    }
    
    double area = 0.0;
    for (size_t i = 1; i < times.size(); ++i) {
        // Convert times from ns to seconds and memories from KB to MB
        double t1 = times[i-1] / 1e9;
        double t2 = times[i] / 1e9;
        double m1 = memories[i-1] / 1000.0;
        double m2 = memories[i] / 1000.0;
        
        // Trapezoidal rule: area += (t2 - t1) * (m1 + m2) / 2
        area += (t2 - t1) * (m1 + m2) / 2.0;
    }
    
    return area;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "0 0 0" << std::endl;
        return 0;
    }
    
    std::string filename = argv[1];
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cout << "0 0 0" << std::endl;
        return 0;
    }
    
    try {
        // Read file content into string
        std::string content((std::istreambuf_iterator<char>(file)), 
                            std::istreambuf_iterator<char>());
        file.close();
        
        // Parse memory profile
        std::vector<long long> times, memories;
        std::istringstream stream(content);
        std::string line;
        long long base_time = -1;
        
        while (std::getline(stream, line)) {
            std::istringstream line_stream(line);
            long long time, memory;
            
            if (line_stream >> time >> memory) {
                if (base_time == -1) {
                    base_time = time;
                }
                times.push_back(time - base_time);
                memories.push_back(memory);
            }
        }
        
        // Calculate statistics
        long long runtime = 0;
        long long max_memory = 0;
        double integral = 0.0;
        
        if (!times.empty()) {
            runtime = times.back();  // Last timestamp is runtime in nanoseconds
            
            // Find maximum memory usage
            if (!memories.empty()) {
                max_memory = *std::max_element(memories.begin(), memories.end());
                
                // Calculate integral
                integral = calculate_efficiency_integral(times, memories);
            }
        }
        
        // Output format: "runtime_ns max_memory_kb integral"
        std::cout << runtime << " " << max_memory << " " << integral << std::endl;
    } catch (...) {
        std::cout << "0 0 0" << std::endl;
    }
    
    return 0;
} 