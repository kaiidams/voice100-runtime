#include <cstdio>
#include <memory>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <world/synthesis.h>
#include <world/codec.h>

int main()
{
    /*
    with open('/home/kaiida/Desktop/sample.dat', 'wb') as f:
    f.write(f0_hat[0].numpy().tobytes())
    f.write(logspc_hat[0].numpy().tobytes())
    f.write(codeap_hat[0].numpy().tobytes())
    */

    /* void Synthesis(const double *f0, int f0_length,
    const double * const *spectrogram, const double * const *aperiodicity,
    int fft_size, double frame_period, int fs, int y_length, double *y)
    */
    std::basic_ifstream<char> ifs("sample.dat", std::ifstream::binary);
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
    std::vector<float> data(data_length);
    ifs.read((char*)data.data(), data_length * sizeof (float));
    if (!ifs) {
        std::cout << "error: only " << ifs.gcount() << " could be read";
        std::cerr << "error" << std::endl;
        std::exit(1);
    }
    ifs.close();

    int f0_length = data_length / (257+1+1);
    std::vector<double> f0(f0_length);
    std::copy(data.begin(), data.begin() + f0_length, f0.begin());

    std::vector<double> spectrogram_data(257 * f0_length);
    for (int i = 0; i < 257 * f0_length; ++i) {
        double v = data[i + f0_length];
        spectrogram_data[i] = std::exp(v);
    }
    //std::copy(data.begin() + f0_length, data.begin() + (1 + 257) * f0_length, spectrogram_data.begin());

    std::vector<double> coded_aperiodicity_data(1 * f0_length);
    std::copy(data.begin() + (1 + 257) * f0_length, data.begin() + (1 + 257 + 1) * f0_length, coded_aperiodicity_data.begin());

    for (auto it = f0.begin(); it != f0.end(); ++it) {
        //std::cout << "f0: " << *it << std::endl;
    }    

    std::vector<double*> spectrogram(f0_length);
    for (int i = 0; i < f0_length; ++i) spectrogram[i] = spectrogram_data.data() + i * 257;
    std::vector<double*> coded_aperiodicity(f0_length);
    for (int i = 0; i < f0_length; ++i) coded_aperiodicity[i] = coded_aperiodicity_data.data() + i;

    int fs = 16000;
    double frame_period = 10.0;
    printf("hello\n");
    int y_length = static_cast<int>((f0_length - 1) *
                                        frame_period / 1000.0 * fs) + 1;
    std::vector<double> y(y_length);

    //std::vector<double> p(257);
    //std::vector<double*> spectrogram(f0_length);
    //std::vector<double*> aperiodicity(f0_length);
    //std::fill(spectrogram.begin(), spectrogram.end(), p.data());
    //std::fill(aperiodicity.begin(), aperiodicity.end(), p.data());

    /*
void DecodeSpectralEnvelope(const double * const *coded_spectral_envelope,
    int f0_length, int fs, int fft_size, int number_of_dimensions,
    double **spectrogram) {
        */
    printf("hello\n");
    std::vector<double*> aperiodicity(f0_length);
    std::vector<double> aperiodicity_data(f0_length * 257);
    for (int i = 0; i < f0_length; ++i) aperiodicity[i] = aperiodicity_data.data() + i * 257;
 
    int fft_size = 512;
    DecodeSpectralEnvelope(coded_aperiodicity.data(), f0_length, fs, fft_size,
      1, aperiodicity.data());

    printf("hello\n");
    Synthesis(
        f0.data(), f0.size(),
        spectrogram.data(),
        aperiodicity.data(),
        fft_size, frame_period, fs,
        y.size(), y.data());

    std::cout << 'y' << y_length << std::endl;

    std::ofstream ofs("test.bin", std::ostream::binary);
    ofs.write((char*)y.data(), y.size() * sizeof (double));
    ofs.close();

    printf("end\n");
}