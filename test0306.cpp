#include "lstm_block_class.hpp"
#include <unordered_map>
#include <ctime>
#include <memory>
using namespace std;

template<unsigned long N>
inline size_t softmax_get_index(const std::array<double, N> &arr, double tau=0.125)
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

template<unsigned long N>
inline size_t get_weighted_random_index(const std::array<double, N> &arr)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    double sum=0.0;
    for(const auto e:arr)
        sum+=e;
    assert(sum<1.0625);
    std::uniform_real_distribution<double> dst(0.0,sum);

    sum=0.0;
    double r=dst(gen);
    for(size_t i=0;i<N;i++)
    {
        sum+=arr[i];
        if(r<=sum) return i;
    }

    assert(0);
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
    const char *asoiaf_filename="asoiaf/asoiaf.txt";

    static constexpr size_t allowed_char_amount=52;
    static constexpr size_t min_training_chars=40;
    static constexpr size_t max_training_chars=1000;
    static constexpr time_t secons_between_saves=60*60;

    static constexpr unsigned long input_size=allowed_char_amount;
    static constexpr unsigned long first_mem_cell_size=200;
    static constexpr unsigned long second_mem_cell_size=100;
    static constexpr unsigned long output_mem_size=allowed_char_amount;
    static constexpr double learning_rate=0.001;

    unique_ptr<LSTMBlock<input_size, first_mem_cell_size>> lstmblock1(new LSTMBlock<input_size, first_mem_cell_size>());
    unique_ptr<LSTMBlock<first_mem_cell_size, second_mem_cell_size>> lstmblock2(new LSTMBlock<first_mem_cell_size, second_mem_cell_size>());
    unique_ptr<SoftmaxBlock<second_mem_cell_size,output_mem_size>> softmaxblock(new SoftmaxBlock<second_mem_cell_size,output_mem_size>);
    lstmblock1->reserve_time_steps(max_training_chars);
    lstmblock2->reserve_time_steps(max_training_chars);
    softmaxblock->reserve_time_steps(max_training_chars);

    //Setup the char_to_index and index_to_char mappings
    const string index_to_char="! \"')(*-,.1032547698;:?]a`cbedgfihkjmlonqpsrutwvyxz~";
    unordered_map<char, size_t> char_to_index;for(size_t i=0;i<index_to_char.size();i++) char_to_index[index_to_char[i]]=i;
    assert(index_to_char.size()==allowed_char_amount and char_to_index.size()==allowed_char_amount);
    //Read the text file
    string asoiaf_content;
    read_file_to_string(asoiaf_filename, asoiaf_content);

    //This matrix will be used as input
    Matrix<1,input_size> X(0.0);
    size_t last_input_index=0;
    //This matrix will be used as "expected output", for training purposes
    Matrix<1,output_mem_size> Y(0.0);
    size_t last_output_index=0;

    //Initialize random number generator and distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dst_start(1,asoiaf_content.size()-max_training_chars-1);
    std::uniform_int_distribution<size_t> dst_lengt(min_training_chars, max_training_chars);
    std::uniform_int_distribution<size_t> dst_rdchar(0, allowed_char_amount-1);

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
                lstmblock1->from_file(in);
                lstmblock2->from_file(in);
                softmaxblock->from_file(in);
            }
        }
    }


    time_t last_time=time(nullptr);
    time_t first_time=last_time;
    //All of the training happens within this loop
    for(;;iteration++)
    {
        //Pick what characters to feed to the lsmts
        size_t asoiaf_start=dst_start(gen);
        size_t asoiaf_length=dst_lengt(gen);
        lstmblock1->set_time_steps(asoiaf_length);
        lstmblock2->set_time_steps(asoiaf_length);
        softmaxblock->set_time_steps(asoiaf_length);

        //Let the network calculate an output given the input
        for(size_t i=0;i<asoiaf_length;i++)
        {
            //Set up input
            X[0][last_input_index]=0.0;
            last_input_index=char_to_index[asoiaf_content[asoiaf_start+i-1]];
            X[0][last_input_index]=1.0;
            lstmblock1->calc(X, i);
            lstmblock2->calc(lstmblock1->get_output(i), i);
            softmaxblock->calc(lstmblock2->get_output(i), i);
        }

        //Calculate deltas
        for(size_t i=asoiaf_length-1;;)
        {
            //Set up output
            Y[0][last_output_index]=0.0;
            last_output_index=char_to_index[asoiaf_content[asoiaf_start+i]];
            Y[0][last_output_index]=1.0;
            softmaxblock->set_first_delta_and_propagate_with_cross_enthropy(Y, lstmblock2->get_delta_output(i), 0);
            // softmaxblock->set_first_delta(Y, i);
            // softmaxblock->propagate_delta(lstmblock2->get_delta_output(i), i);
            lstmblock2->propagate_delta(lstmblock1->get_delta_output(i), i, asoiaf_length);
            lstmblock1->propagate_delta(i, asoiaf_length);
            if(i--==0)break;
        }

        //Update weights
        for(size_t i=0;i<asoiaf_length;i++)
        {
            //Set up input
            X[0][last_input_index]=0.0;
            last_input_index=char_to_index[asoiaf_content[asoiaf_start+i-1]];
            X[0][last_input_index]=1.0;
            lstmblock1->update_weights(X, i, learning_rate);
            lstmblock2->update_weights(lstmblock1->get_output(i), i, learning_rate);
            softmaxblock->update_weights(lstmblock2->get_output(i), i, learning_rate);
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
                lstmblock1->to_file(out);
                lstmblock2->to_file(out);
                softmaxblock->to_file(out);
            }
            //Save some data and some output examples to file
            sprintf(cbuffer, output_filename, save_counter);
            {
                ofstream out(cbuffer,std::ios_base::trunc);
                assert(out.good());
                out << "Current iteration: " << iteration << endl;
                out << "Seconds elapsed since program started: " << time(nullptr)-first_time << endl;
                out << "Used memory cells: " << first_mem_cell_size << ", " << second_mem_cell_size << endl;
                out << endl;
                out << endl;
                out << endl;
                out << endl;
                for(size_t ex=0;ex<10;ex++)
                {
                    out << "################################ EXAMPLE " << ex+1 << " ################################" << endl;
                    size_t next_input_index=dst_rdchar(gen);
                    lstmblock1->set_time_steps(2000);
                    lstmblock2->set_time_steps(2000);
                    softmaxblock->set_time_steps(2000);
                    for(size_t i=0;i<2000;i++)
                    {
                        X[0][last_input_index]=0.0;
                        last_input_index=next_input_index;
                        X[0][last_input_index]=1.0;
                        lstmblock1->calc(X, i);
                        lstmblock2->calc(lstmblock1->get_output(i), i);
                        softmaxblock->calc(lstmblock2->get_output(i), i);
                        next_input_index=get_weighted_random_index(softmaxblock->get_output(i)[0]);
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