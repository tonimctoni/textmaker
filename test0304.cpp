#include "lstm_blocks.hpp"
#include <unordered_map>
#include <ctime>
#include <memory>
using namespace std;

template<unsigned long N>
inline size_t softmax_get_index(std::array<double, N> &arr, double tau=0.125)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dst(0.0,1.0);
    std::array<double, N> probs;
    double sum=0;
    double r=dst(gen);

    for(size_t i=0;i<N;i++)
    {
        probs[i]=exp(arr[i]/tau);
        sum+=probs[i];
    }

    for(size_t i=0;i<N;i++)
    {
        probs[i]/=sum;
        if(i!=0)probs[i]+=probs[i-1];
        if(r<=probs[i]) return i;
    }

    assert(0);
    // return N-1;
}

inline void read_file_to_string(const char *filename, string &out_str)
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    assert(in.good());
    in.seekg(0, std::ios::end);
    out_str.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(&out_str[0], out_str.size());
    in.close();
}

inline void read_file_to_string(const char *filename, string &out_str, size_t max_size)
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    assert(in.good());
    in.seekg(0, std::ios::end);
    out_str.resize(size_t(in.tellg())>max_size?max_size:size_t(in.tellg()));
    in.seekg(0, std::ios::beg);
    in.read(&out_str[0], size_t(in.tellg())>max_size?max_size:size_t(in.tellg()));
    in.close();
}


//remove asserts
int main()
{
    const char *savecounter_filename="data/savecounter.svc";
    const char *savestate_filename="data/savestate%08lu.sst";
    const char *output_filename="data/output%08lu.sst";
    const char *css_filename="css/css%05lu.css";

    static constexpr size_t allowed_char_amount=98;
    static constexpr size_t css_file_amount=14985;
    static constexpr size_t max_filesize=15000;//1669791
    static constexpr time_t secons_between_saves=60*60;

    static constexpr unsigned long input_size=allowed_char_amount;
    static constexpr unsigned long mem_cell_size=500;
    static constexpr unsigned long output_size=allowed_char_amount;
    static constexpr double learning_rate=0.001;

    unique_ptr<LstmParam<input_size, mem_cell_size>> lstm_params(new LstmParam<input_size, mem_cell_size>);
    vector<LstmState<input_size, mem_cell_size>> lstm_states;
    unique_ptr<LayerParam<mem_cell_size,output_size>> layer_params(new LayerParam<mem_cell_size,output_size>);
    vector<LayerState<mem_cell_size, output_size>> layer_states;
    lstm_states.reserve(max_filesize);
    layer_states.reserve(max_filesize);

    //Setup the char_to_index and index_to_char mappings
    const string index_to_char="abcdefghijklmnopqrstuvwxzyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\n\t\r \"\'(){}[]+-*/.,:;_@#%$!?=\\<>~^|&`";
    unordered_map<char, size_t> char_to_index;for(size_t i=0;i<index_to_char.size();i++) char_to_index[index_to_char[i]]=i;
    assert(index_to_char.size()==allowed_char_amount and char_to_index.size()==allowed_char_amount);
    //Read all css files. For this, the constant css_file_amount has to be set correctly.
    vector<string> css_files;
    {
        char buffer[256];
        css_files.resize(css_file_amount);
        for(size_t i=0;i<css_file_amount;i++)
        {
            sprintf(buffer, css_filename, i+1);
            read_file_to_string(buffer, css_files[i], max_filesize);
            assert(css_files[i].size()!=0);
        }
    }

    //This matrix will be used as input
    Matrix<1,input_size> X(0.0);
    size_t last_input_index=0;
    //This matrix will be used as "expected output", for training purposes
    Matrix<1,output_size> Y(0.0);
    size_t last_output_index=0;

    //Initialize random number generator and distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dst(0,css_file_amount-1);

    //Retrieve the number of the last saved state and last iteration (if exists) and the saved state with this number
    size_t save_counter=0;
    size_t iteration=0;
    {
        ifstream in(savecounter_filename);
        if(in.good())
        {
            in >> save_counter >> iteration;
            if(save_counter!=0)
            {
                char cbuffer[256];
                sprintf(cbuffer, savestate_filename, save_counter);
                ifstream in(cbuffer);
                assert(in.good());
                lstm_params->from_file(in);
                layer_params->from_file(in);
            }
        }
    }


    time_t last_time=time(nullptr);
    time_t first_time=last_time;
    //All of the training happens within this loop
    for(;;iteration++)
    {
        //Pick a css file at random, and then for each character...
        string &css_file=css_files[dst(gen)];
        lstm_states.resize(css_file.size());
        layer_states.resize(css_file.size());

        //Let the network calculate an output given the input
        for(size_t i=0;i<css_file.size();i++)
        {
            //Set up input
            X[0][last_input_index]=0.0;
            if(i!=0) //In the first round the first char should be predicted without input
            {
                last_input_index=char_to_index[css_file[i-1]];
                X[0][last_input_index]=1.0;
            }
            calc_lstm(X, lstm_states, *lstm_params, i);
            calc_layer_tanh(lstm_states[i].state_h, layer_states, *layer_params, i);
        }

        //Calculate deltas
        for(size_t i=css_file.size()-1;;)
        {
            //Set up output
            Y[0][last_output_index]=0.0;
            last_output_index=char_to_index[css_file[i]];
            Y[0][last_output_index]=1.0;
            // set_first_delta_and_propagate_layer_tanh(Y, lstm_states[i].delta_h, layer_states, *layer_params, i);
            set_first_delta_layer_tanh(Y, layer_states, i);
            propagate_delta_layer_tanh(lstm_states[i].delta_h, layer_states, *layer_params, i);
            propagate_delta_lstm(lstm_states, *lstm_params, i, css_file.size());
            if(i--==0)break;
        }

        //Update weights
        for(size_t i=0;i<css_file.size();i++)
        {
            //Set up input
            X[0][last_input_index]=0.0;
            if(i!=0) //In the first round the first char should be predicted without input
            {
                last_input_index=char_to_index[css_file[i-1]];
                X[0][last_input_index]=1.0;
            }
            update_weights_lstm(X, lstm_states, *lstm_params, i, learning_rate);
            update_weights_layer_tanh(lstm_states[i].state_h, layer_states, *layer_params, i, learning_rate);
        }


        if(time(nullptr)-last_time>secons_between_saves)
        {
            last_time=time(nullptr);
            print("Saving current state...");
            save_counter++;
            char cbuffer[256];
            //Save weights to file
            sprintf(cbuffer, savestate_filename, save_counter);
            {
                ofstream out(cbuffer,std::ios_base::trunc);
                assert(out.good());
                lstm_params->to_file(out);
                layer_params->to_file(out);
            }
            //Save some data and some output examples to file
            sprintf(cbuffer, output_filename, save_counter);
            {
                ofstream out(cbuffer,std::ios_base::trunc);
                assert(out.good());
                out << "Current iteration: " << iteration << endl;
                out << "Seconds elapsed since program started: " << time(nullptr)-first_time << endl;
                out << "Used memory cells: " << mem_cell_size << endl;
                out << endl;
                out << endl;
                out << endl;
                out << endl;
                for(size_t ex=0;ex<10;ex++)
                {
                    out << "################################ EXAMPLE " << ex+1 << " ################################" << endl;
                    size_t next_input_index=0;
                    lstm_states.resize(2000);
                    layer_states.resize(2000);
                    for(size_t i=0;i<2000;i++)
                    {
                        X[0][last_input_index]=0.0;
                        if(i!=0)
                        {
                            last_input_index=next_input_index;
                            X[0][last_input_index]=1.0;
                        }
                        calc_lstm(X, lstm_states, *lstm_params, i);
                        calc_layer_tanh(lstm_states[i].state_h, layer_states, *layer_params, i);
                        next_input_index=softmax_get_index(layer_states[i].output[0]);
                        out << index_to_char[next_input_index];
                    }
                    out << endl;
                    out << endl;
                    out << endl;
                    out << endl;
                }
            }
            {
                ofstream out(savecounter_filename,std::ios_base::trunc);
                assert(out.good());
                out << save_counter << "\t" << iteration << endl;
            }
            print("State with number", save_counter, "saved");
        }
        print("Iteration:", iteration);
    }
    return 0;
}