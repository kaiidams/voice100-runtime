#include <cstdio>
#include <memory>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>

extern "C" int Voice100VocoderDecode(
    const float* f0, const float* logspc, const float* codedap, int f0_length,
    int fft_size, double frame_period, int fs, float log_offset, int16_t* y, int y_length);
std::vector<float> read_data(const char* path);
void write_data(const std::vector<int16_t>& y);

int main()
{
    printf("start\n");
    /*
    void DecodeSpectralEnvelope(const double * const *coded_spectral_envelope,
    int f0_length, int fs, int fft_size, int number_of_dimensions,
    double **spectrogram) {
    */

    /* void Synthesis(const double *f0, int f0_length,
    const double * const *spectrogram, const double * const *aperiodicity,
    int fft_size, double frame_period, int fs, int y_length, double *y)
    */
    auto data = read_data("sample.dat");
    int fs = 16000;
    int fft_size = 512;
    double frame_period = 10.0;
    int spectrogram_dim = fft_size / 2 + 1;
    int coded_aperiodicity_dim = 1;

    int f0_length = data.size() / (1 + spectrogram_dim + coded_aperiodicity_dim);
    float log_offset = 1e-15;

    // Decode
    float* f0 = data.data();
    float* logspc = data.data() + f0_length;
    float* codedap = data.data() + (1 + spectrogram_dim) * f0_length;
    int y_length = Voice100VocoderDecode(
        f0, logspc, codedap, f0_length,
        fft_size, frame_period, fs, log_offset, nullptr, 0);

    std::vector<int16_t> y(y_length);
    Voice100VocoderDecode(
        f0, logspc, codedap, f0_length,
        fft_size, frame_period, fs, log_offset, y.data(), y_length);

    std::cout << "y_length: " << y_length << std::endl;

    printf("end\n");
    write_data(y);
}

std::vector<float> read_data(const char *path)
{
    std::basic_ifstream<char> ifs(path, std::ifstream::binary);
    if (!ifs) {
        std::cerr << "error" << std::endl;
        std::exit(1);
    }
    ifs.seekg(0, ifs.end);
    int data_length = ifs.tellg() / sizeof (float);
    std::cout << data_length << std::endl;
    std::cout << (double)data_length / (257+1+1) << std::endl;
    ifs.seekg(0, ifs.beg);
    //int data_length = 2000;
    std::vector<float> float_data(data_length);
    ifs.read((char*)float_data.data(), data_length * sizeof (float));
    if (!ifs) {
        std::cout << "error: only " << ifs.gcount() << " could be read";
        std::cerr << "error" << std::endl;
        std::exit(1);
    }
    ifs.close();
    return float_data;
#if 0
    std::vector<double> data(data_length);
    std::copy(float_data.begin(), float_data.end(), data.begin());
    return data;
#endif
}

void write_data(const std::vector<int16_t>& y)
{
    std::ofstream ofs("output.dat", std::ostream::binary);
    ofs.write((char*)y.data(), y.size() * sizeof (double));
    ofs.close();
}
