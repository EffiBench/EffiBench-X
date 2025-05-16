#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <cstring>

static long getRSSkB(pid_t pid) {
    std::string statmPath = "/proc/" + std::to_string(pid) + "/statm";
    std::ifstream statmFile(statmPath);
    if (!statmFile.is_open()) return 0;

    long size = 0, resident = 0;
    statmFile >> size >> resident;
    statmFile.close();

    static const long pageSize_kB = sysconf(_SC_PAGESIZE) / 1024;
    return resident * pageSize_kB;
}

static long long nowNs() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <target_interval_seconds> <log_file> [<memory_limit_mb>] -- <command> [args...]\n"
                  << "Memory limit defaults to 0 (no limit) if not specified.\n";
        return 1;
    }

    // Desired interval in seconds (e.g., 0.001 for 1 millisecond)
    double targetIntervalSeconds = std::stod(argv[1]);
    long long targetIntervalNs = static_cast<long long>(targetIntervalSeconds * 1e9);
    std::string logFileName = argv[2];
    
    // Memory limit in MB - special value of 0 means no memory limit
    long memoryLimitMB = 0; // Default: no memory limit
    bool enforceMemoryLimit = false;
    
    // Variables for argument processing
    int argOffset = 0;
    
    // Check if memory limit is provided
    if (argc > 4 && std::strcmp(argv[3], "--") != 0) {
        memoryLimitMB = std::stol(argv[3]);
        enforceMemoryLimit = (memoryLimitMB > 0);
        argOffset = 1;
    }

    // Parse the child command after "--"
    int separatorIndex = -1;
    for (int i = 3 + argOffset; i < argc; ++i) {
        if (std::strcmp(argv[i], "--") == 0) {
            separatorIndex = i;
            break;
        }
    }
    if (separatorIndex < 0 || separatorIndex == argc - 1) {
        std::cerr << "Error: Missing '-- <command> [args...]' part.\n";
        return 1;
    }

    // Prepare child args
    std::vector<char*> childArgs;
    for (int i = separatorIndex + 1; i < argc; ++i)
        childArgs.push_back(argv[i]);
    childArgs.push_back(nullptr);

    pid_t childPid = fork();
    if (childPid < 0) {
        perror("fork");
        return 1;
    }

    if (childPid == 0) {
        // Child process
        execvp(childArgs[0], childArgs.data());
        perror("execvp");
        _exit(1);
    }

    // Parent process
    std::vector<std::pair<long long, long>> samples;
    samples.reserve(100000);

    long long previousSampleTime = 0;
    long long nextSampleTime = 0;
    
    // Convert MB to KB only if we have a memory limit
    long memoryLimitKB = enforceMemoryLimit ? memoryLimitMB * 1024 : 0;

    // Start measurement loop
    previousSampleTime = nowNs();
    nextSampleTime = previousSampleTime + targetIntervalNs;

    int child_exit_status = 0; // Variable to store the child's exit status

    while (true) {
        int status = 0;
        pid_t ret = waitpid(childPid, &status, WNOHANG);
        if (ret < 0) {
            // Error or child already gone
            break;
        }
        if (ret > 0) {
            // Child exited
            if (WIFEXITED(status)) {
                child_exit_status = WEXITSTATUS(status);
            } else if (WIFSIGNALED(status)) {
                child_exit_status = 128 + WTERMSIG(status); // Standard convention
            }
            break;
        }

        // Measure memory
        long long timestampNs = nowNs();
        long rss_kB = getRSSkB(childPid);
        
        // Check if memory exceeds the limit (only if a limit is set)
        if (enforceMemoryLimit && rss_kB > memoryLimitKB) {
            // Process is using more than the memory limit, kill it
            std::cerr << "Process exceeded memory limit (" << memoryLimitMB << " MB), killing..." << std::endl;
            kill(childPid, SIGKILL);
            int kill_status;
            waitpid(childPid, &kill_status, 0);  // Wait for the process to end
            child_exit_status = 137; // Set exit code for OOM

            // Write samples to the log file before exiting
            std::ofstream outFile(logFileName);
            if (!outFile.is_open()) {
                std::cerr << "Error opening log file: " << logFileName << "\n";
                return 1;
            }
            for (auto& s : samples) {
                outFile << s.first << " " << s.second << "\n";
            }
            outFile.close();
            
            return child_exit_status;  // Return the OOM exit code
        }
        
        samples.emplace_back(timestampNs, rss_kB);

        // Update next sample time
        nextSampleTime += targetIntervalNs;

        // Sleep or spin until we reach nextSampleTime
        // (using sleep_until for simplicity, though it may not be precise to 1 µs)
        auto tp = std::chrono::steady_clock::time_point(std::chrono::nanoseconds(nextSampleTime));
        std::this_thread::sleep_until(tp);
    }

    // Child has exited -> flush samples
    std::ofstream outFile(logFileName);
    if (!outFile.is_open()) {
        std::cerr << "Error opening log file: " << logFileName << "\n";
        return 1;
    }
    for (auto& s : samples) {
        outFile << s.first << " " << s.second << "\n";
    }
    outFile.close();

    return child_exit_status; // Return the actual exit status of the child
}
