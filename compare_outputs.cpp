#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << "first_output second_output" << std::endl;
        return 1;
    }
    std::ifstream fst_file(argv[1]);
    std::ifstream snd_file(argv[2]);
    double fst;
    double snd;
    double res = 0;
    int counter;
    while (fst_file.good() && snd_file.good()) {
        fst_file >> fst;
        snd_file >> snd;
        res += (fst - snd) * (fst - snd);
        ++counter;
    }
    if (fst_file.good() || snd_file.good()) {
        std::cout << "Reached end only in one file" << std::endl;
        return 1;
    }
    std::cout << "Difference is " << (res / counter) << std::endl;
    if (res / counter < 1e-6) {
        std::cout << "Test passed" << std::endl;
    } else {
        std::cout << "Test failed" << std::endl;
    }
    return 0;
}
