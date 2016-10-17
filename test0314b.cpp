#include "momented_lstm_block_class.hpp"
#include <unordered_map>
#include <ctime>
#include <memory>
using namespace std;

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

template<unsigned long mat_size>
class OneHot
{
private:
    size_t hot_index;
    Matrix<1, mat_size> X;
public:
    OneHot()noexcept:hot_index(0), X(0.0)
    {
    }

    inline void set(size_t index) noexcept
    {
        assert(index<mat_size);
        X[0][hot_index]=0.0;
        hot_index=index;
        X[0][hot_index]=1.0;
    }

    inline void reset() noexcept
    {
        X[0][hot_index]=0.0;
    }

    inline const Matrix<1, mat_size>& get() const noexcept
    {
        return X;
    }
};

struct MyNet
{
    unique_ptr<TahnPerceptronBlock<46,11>> perceptronblock;
    unique_ptr<LSTMBlock<11, 400>> lstmblock1;
    unique_ptr<LSTMBlock<400, 200>> lstmblock2;
    unique_ptr<LSTMBlock<200, 100>> lstmblock3;

    MyNet(const char *filename):
    perceptronblock(new TahnPerceptronBlock<46,11>),
    lstmblock1(new LSTMBlock<11, 400>()),
    lstmblock2(new LSTMBlock<400, 200>()),
    lstmblock3(new LSTMBlock<200, 100>())
    {
        ifstream in(filename, std::ios::binary);
        assert(in.good());
        perceptronblock->from_bin_file(in);
        lstmblock1->from_bin_file(in);
        lstmblock2->from_bin_file(in);
        lstmblock3->from_bin_file(in);
    }

    inline void reserve_time_steps(size_t ts) noexcept
    {
        perceptronblock->reserve_time_steps(ts);
        lstmblock1->reserve_time_steps(ts);
        lstmblock2->reserve_time_steps(ts);
        lstmblock3->reserve_time_steps(ts);
    }

    inline void set_time_steps(size_t ts) noexcept
    {
        perceptronblock->set_time_steps(ts);
        lstmblock1->set_time_steps(ts);
        lstmblock2->set_time_steps(ts);
        lstmblock3->set_time_steps(ts);
    }

    inline void calc(const OneHot<46> &X, size_t ts) noexcept
    {
        perceptronblock->calc(X.get(), ts);
        lstmblock1->calc(perceptronblock->get_output(ts), ts);
        lstmblock2->calc(lstmblock1->get_output(ts), ts);
        lstmblock3->calc(lstmblock2->get_output(ts), ts);
    }

    inline const Matrix<1,100>& get_output(size_t ts) const noexcept
    {
        return lstmblock3->get_output(ts);
    }
};

//remove asserts
int main()
{
    const char *savecounter_filename="data/savecounter.svc";
    const char *savestate_filename="data/savestate%08lu.sst";
    const char *output_filename="data/output%08lu.sst";
    const char *asoiaf_filename="../asoiaf/asoiaf.txt";
    static constexpr size_t output_length=2000;

    // static constexpr size_t allowed_char_amount=46;
    static constexpr size_t min_training_chars=40;
    static constexpr size_t max_training_chars=1000;
    static constexpr time_t secons_between_saves=60*60;

    // static constexpr unsigned long input_size=allowed_char_amount;
    // static constexpr unsigned long reduced_input_size=allowed_char_amount/4;
    // static constexpr unsigned long first_mem_cell_size=400;
    // static constexpr unsigned long second_mem_cell_size=200;
    // static constexpr unsigned long third_mem_cell_size=100;
    // static constexpr unsigned long output_mem_size=allowed_char_amount;
    double learning_rate=0.01;
    double momentum=0.5;
    static constexpr size_t batch_size=1;

    // unique_ptr<TahnPerceptronBlock<input_size,reduced_input_size>> perceptronblock(new TahnPerceptronBlock<input_size,reduced_input_size>);
    // unique_ptr<LSTMBlock<reduced_input_size, first_mem_cell_size>> lstmblock1(new LSTMBlock<reduced_input_size, first_mem_cell_size>());
    // unique_ptr<LSTMBlock<first_mem_cell_size, second_mem_cell_size>> lstmblock2(new LSTMBlock<first_mem_cell_size, second_mem_cell_size>());
    // unique_ptr<LSTMBlock<second_mem_cell_size, third_mem_cell_size>> lstmblock3(new LSTMBlock<second_mem_cell_size, third_mem_cell_size>());
    // unique_ptr<SoftmaxBlock<third_mem_cell_size,output_mem_size>> softmaxblock(new SoftmaxBlock<third_mem_cell_size,output_mem_size>);
    // perceptronblock->reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);
    // lstmblock1->reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);
    // lstmblock2->reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);
    // lstmblock3->reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);
    // softmaxblock->reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);

    MyNet myneta("data/a.sst");
    MyNet mynetb("data/b.sst");
    MyNet mynetc("data/c.sst");
    MyNet mynetd("data/d.sst");
    vector<Matrix<1, 400>> inter_matrixes;
    unique_ptr<TahnPerceptronBlock<400,300>> perceptronblock1(new TahnPerceptronBlock<400,300>);
    unique_ptr<TahnPerceptronBlock<300,100>> perceptronblock2(new TahnPerceptronBlock<300,100>);
    unique_ptr<SoftmaxBlock<100,46>> softmaxblock(new SoftmaxBlock<100,46>);
    myneta.reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);
    mynetb.reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);
    mynetc.reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);
    mynetd.reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);
    inter_matrixes.reserve(max_training_chars>output_length?max_training_chars:output_length);
    perceptronblock1->reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);
    perceptronblock2->reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);
    softmaxblock->reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);

    //Setup the char_to_index and index_to_char mappings
    const string index_to_char="! ')(-,.103254769;:?acbedgfihkjmlonqpsrutwvyxz";
    unordered_map<char, size_t> char_to_index;for(size_t i=0;i<index_to_char.size();i++) char_to_index[index_to_char[i]]=i;
    assert(index_to_char.size()==46 and char_to_index.size()==46);
    //Read the text file
    string asoiaf_content;
    read_file_to_string(asoiaf_filename, asoiaf_content);

    // //This matrix will be used as input
    OneHot<46> X;
    OneHot<46> Y;

    //Initialize random number generator and distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dst_start(1,asoiaf_content.size()-max_training_chars-1);
    std::uniform_int_distribution<size_t> dst_lengt(min_training_chars, max_training_chars);
    std::uniform_int_distribution<size_t> dst_rdchar(0, 46-1);

    size_t save_counter=0;
    size_t iteration=0;
    ifstream in(savecounter_filename, std::ios::binary);
    if(in.good())
    {
        in >> save_counter >> iteration;
        if(save_counter!=0)
        {
            char cbuffer[256];
            sprintf(cbuffer, savestate_filename, save_counter);
            ifstream in(cbuffer);
            assert(in.good());
            perceptronblock1->from_bin_file(in);
            perceptronblock2->from_bin_file(in);
            softmaxblock->from_bin_file(in);
        }
    }

    time_t last_time=time(nullptr);
    time_t first_time=last_time;
    //All of the training happens within this loop
    for(;;iteration++)
    {
        if(iteration%1000==0)
        {
            learning_rate=0.01*pow(.955, (iteration/1000));// gets divided by 10 every 50k steps
            if(iteration<=50000) momentum=0.5+0.008*(iteration/1000);
            else if(iteration<=100000)momentum=0.9+0.0018*((iteration-50000)/1000);
            else momentum=.99;
        }
        perceptronblock1->apply_momentum(momentum);
        perceptronblock2->apply_momentum(momentum);
        softmaxblock->apply_momentum(momentum);
        for(size_t batch=0;batch<batch_size;batch++)
        {
            //Pick what characters to feed to the lsmts
            size_t asoiaf_start=dst_start(gen);
            size_t asoiaf_length=dst_lengt(gen);
            myneta.set_time_steps(asoiaf_length);
            mynetb.set_time_steps(asoiaf_length);
            mynetc.set_time_steps(asoiaf_length);
            mynetd.set_time_steps(asoiaf_length);
            inter_matrixes.resize(asoiaf_length);
            perceptronblock1->set_time_steps(asoiaf_length);
            perceptronblock2->set_time_steps(asoiaf_length);
            softmaxblock->set_time_steps(asoiaf_length);

            //Let the network calculate an output given the input
            for(size_t i=0;i<asoiaf_length;i++)
            {
                //Set up input
                X.set(char_to_index[asoiaf_content[asoiaf_start+i-1]]);

                myneta.calc(X, i);
                mynetb.calc(X, i);
                mynetc.calc(X, i);
                mynetd.calc(X, i);
                inter_matrixes[i].set_from_four(myneta.get_output(i), mynetb.get_output(i), mynetc.get_output(i), mynetd.get_output(i));
                perceptronblock1->calc(inter_matrixes[i], i);
                perceptronblock2->calc(perceptronblock1->get_output(i), i);
                softmaxblock->calc(perceptronblock2->get_output(i), i);
            }

            //Calculate deltas
            for(size_t i=asoiaf_length-1;;)
            {
                //Set up output
                Y.set(char_to_index[asoiaf_content[asoiaf_start+i]]);
                softmaxblock->set_first_delta_and_propagate_with_cross_enthropy(Y.get(), perceptronblock2->get_delta_output(i), i);
                perceptronblock2->propagate_delta(perceptronblock1->get_delta_output(i), i);
                perceptronblock1->propagate_delta(i);

                if(i--==0)break;
            }

            //Update gradients
            for(size_t i=0;i<asoiaf_length;i++)
            {
                perceptronblock1->accumulate_gradients(inter_matrixes[i], i);
                perceptronblock2->accumulate_gradients(perceptronblock1->get_output(i), i);
                softmaxblock->accumulate_gradients(perceptronblock2->get_output(i), i);
            }

            perceptronblock1->update_weights_momentum(learning_rate);
            perceptronblock2->update_weights_momentum(learning_rate);
            softmaxblock->update_weights_momentum(learning_rate);
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
                ofstream out(cbuffer,std::ios_base::trunc|std::ios::binary);
                assert(out.good());
                // perceptronblock->to_bin_file(out);
                // lstmblock1->to_bin_file(out);
                // lstmblock2->to_bin_file(out);
                // lstmblock3->to_bin_file(out);
                perceptronblock1->to_bin_file(out);
                perceptronblock2->to_bin_file(out);
                softmaxblock->to_bin_file(out);
            }
            //Save some data and some output examples to file
            sprintf(cbuffer, output_filename, save_counter);
            {
                ofstream out(cbuffer,std::ios_base::trunc);
                assert(out.good());
                out << "Current iteration: " << iteration << endl;
                out << "Seconds elapsed since program started: " << time(nullptr)-first_time << endl;
                // out << "Used memory cells: " << first_mem_cell_size << ", " << second_mem_cell_size << ", " << third_mem_cell_size << endl;
                // out << "Batch size: " << batch_size << endl;
                out << "Learning rate: " << learning_rate << endl;
                out << "Momentum: " << momentum << endl;
                out << endl;
                out << endl;
                out << endl;
                out << endl;
                for(size_t ex=0;ex<10;ex++)
                {
                    out << "################################ EXAMPLE " << ex+1 << " ################################" << endl;
                    size_t next_input_index=dst_rdchar(gen);
                    myneta.set_time_steps(output_length);
                    mynetb.set_time_steps(output_length);
                    mynetc.set_time_steps(output_length);
                    mynetd.set_time_steps(output_length);
                    inter_matrixes.resize(output_length);
                    perceptronblock1->set_time_steps(output_length);
                    perceptronblock2->set_time_steps(output_length);
                    softmaxblock->set_time_steps(output_length);
                    for(size_t i=0;i<output_length;i++)
                    {
                        X.set(next_input_index);

                        myneta.calc(X, i);
                        mynetb.calc(X, i);
                        mynetc.calc(X, i);
                        mynetd.calc(X, i);
                        inter_matrixes[i].set_from_four(myneta.get_output(i), mynetb.get_output(i), mynetc.get_output(i), mynetd.get_output(i));
                        perceptronblock1->calc(inter_matrixes[i], i);
                        perceptronblock2->calc(perceptronblock1->get_output(i), i);
                        softmaxblock->calc(perceptronblock2->get_output(i), i);
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