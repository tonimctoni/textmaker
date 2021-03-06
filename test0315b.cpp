#include "perceptron_timeseries_class.hpp"
#include "LSTM_class.hpp"
#include "softmax_timeseries_class.hpp"
#include <unordered_map>
// #include <ctime>
#include <memory>
using namespace std;

template<unsigned long first_mem_cell_size, unsigned long second_mem_cell_size, unsigned long third_mem_cell_size>
class MyNet
{
private:
    static constexpr size_t allowed_char_amount=46;
    static constexpr unsigned long input_size=allowed_char_amount;
    static constexpr unsigned long reduced_input_size=allowed_char_amount/4;
    static constexpr unsigned long output_mem_size=allowed_char_amount;

    using Block01=BaseTahnPerceptronBlock<input_size,reduced_input_size>;
    using Block02=BaseLSTMBlock<reduced_input_size, first_mem_cell_size>;
    using Block03=BaseLSTMBlock<first_mem_cell_size, second_mem_cell_size>;
    using Block04=BaseLSTMBlock<second_mem_cell_size, third_mem_cell_size>;
    using Block05=BaseSoftmaxBlock<third_mem_cell_size, output_mem_size>;

    unique_ptr<Block01> perceptronblock;
    unique_ptr<Block02> lstmblock1;
    unique_ptr<Block03> lstmblock2;
    unique_ptr<Block04> lstmblock3;
    unique_ptr<Block05> softmaxblock;
public:
    MyNet(const char *filename)noexcept: perceptronblock(new Block01), lstmblock1(new Block02), lstmblock2(new Block03), lstmblock3(new Block04), softmaxblock(new Block05)
    {
        ifstream in(filename);
        assert(in.good());
        perceptronblock->only_wb_from_bin_file(in);
        lstmblock1->only_wb_from_bin_file(in);
        lstmblock2->only_wb_from_bin_file(in);
        lstmblock3->only_wb_from_bin_file(in);
        softmaxblock->only_wb_from_bin_file(in);
    }

    inline void set_time_steps(size_t time_steps) noexcept
    {
        perceptronblock->set_time_steps(time_steps);
        lstmblock1->set_time_steps(time_steps);
        lstmblock2->set_time_steps(time_steps);
        lstmblock3->set_time_steps(time_steps);
        softmaxblock->set_time_steps(time_steps);
    }

    inline const Matrix<1,output_mem_size>& calc(const Matrix<1,input_size> &X, size_t time_step) noexcept
    {
        perceptronblock->calc(X, time_step);
        lstmblock1->calc(perceptronblock->get_output(time_step), time_step);
        lstmblock2->calc(lstmblock1->get_output(time_step), time_step);
        lstmblock3->calc(lstmblock2->get_output(time_step), time_step);
        softmaxblock->calc(lstmblock3->get_output(time_step), time_step);
        return softmaxblock->get_output(time_step);
    }
};

int main()
{
    MyNet<512,256,128> mynetn("outs/tn.wab");
    MyNet<512,256,128> mynetr("outs/tr.wab");

    static constexpr size_t allowed_char_amount=46;
    const string index_to_char="! ')(-,.103254769;:?acbedgfihkjmlonqpsrutwvyxz";
    unordered_map<char, size_t> char_to_index;for(size_t i=0;i<index_to_char.size();i++) char_to_index[index_to_char[i]]=i;
    assert(index_to_char.size()==allowed_char_amount and char_to_index.size()==allowed_char_amount);

    static constexpr unsigned long input_size=allowed_char_amount;
    static constexpr unsigned long output_mem_size=allowed_char_amount;


    static constexpr size_t output_size=125;
    string start_string;
    string end_stringn;
    string end_stringr;
    // string end_stringc;
    string end_strings;
    string end_stringm;
    print("Starting string:");
    getline(cin, start_string);
    OneHot<input_size> X;
    mynetn.set_time_steps(start_string.size()+output_size);
    mynetr.set_time_steps(start_string.size()+output_size);

    end_stringn.reserve(output_size);
    end_stringr.reserve(output_size);
    end_strings.reserve(output_size);
    end_stringm.reserve(output_size);
    for(size_t i=0;i<start_string.size();i++)
    {
        if(i<start_string.size()-1)
        {
            X.set(char_to_index.at(start_string[i]));
            mynetn.calc(X.get(), i);
            mynetr.calc(X.get(), i);
        }
        else
        {
            X.set(char_to_index.at(start_string[i]));
            end_stringn.push_back(index_to_char.at(get_weighted_random_index(mynetn.calc(X.get(), i)[0])));
            end_stringr.push_back(index_to_char.at(get_weighted_random_index(mynetr.calc(X.get(), i)[0])));
        }
    }

    for(size_t i=start_string.size();i<start_string.size()+output_size-1;i++)
    {
        X.set(char_to_index.at(end_stringn.back()));
        end_stringn.push_back(index_to_char.at(get_weighted_random_index(mynetn.calc(X.get(), i)[0])));
        X.set(char_to_index.at(end_stringr.back()));
        end_stringr.push_back(index_to_char.at(get_weighted_random_index(mynetr.calc(X.get(), i)[0])));
    }


    // for(size_t i=0;i<start_string.size()-1;i++)
    // {
    //     X.set(char_to_index.at(start_string[i]));
    //     mynetn.calc(X.get(), i);
    //     mynetr.calc(X.get(), i);
    // }
    // X.set(char_to_index.at(start_string.back()));
    // auto& outn=mynetn.calc(X.get(), start_string.size()-1);
    // auto& outr=mynetr.calc(X.get(), start_string.size()-1);
    // Matrix<1,output_mem_size> output;
    // output.set(0.0);
    // output.add(outn);
    // output.add(outr);
    // for(size_t j=0;j<output_mem_size;j++) if(outn[0][j]<.01 or outr[0][j]<.01) output[0][j]=.0;
    // if(output.sum()==0)
    // {
    //     output.set(outn);
    //     output.mul(outr);
    //     cout << "(**)";
    // }
    // output.div(output.sum());
    // end_stringc.push_back(index_to_char.at(get_weighted_random_index(output[0])));

    // for(size_t i=start_string.size();i<start_string.size()+output_size-1;i++)
    // {
    //     X.set(char_to_index.at(end_stringc.back()));
    //     auto& outn=mynetn.calc(X.get(), i);
    //     auto& outr=mynetr.calc(X.get(), i);
    //     output.set(0.0);
    //     output.add(outn);
    //     output.add(outr);
    //     for(size_t j=0;j<output_mem_size;j++) if(outn[0][j]<.01 or outr[0][j]<.01) output[0][j]=.0;
    //     if(output.sum()==0)
    //     {
    //         output.set(outn);
    //         output.mul(outr);
    //         cout << "(**)";
    //     }
    //     output.div(output.sum());
    //     end_stringc.push_back(index_to_char.at(get_weighted_random_index(output[0])));
    // }

    {
        for(size_t i=0;i<start_string.size()-1;i++)
        {
            X.set(char_to_index.at(start_string[i]));
            mynetn.calc(X.get(), i);
            mynetr.calc(X.get(), i);
        }
        X.set(char_to_index.at(start_string.back()));
        Matrix<1,output_mem_size> output;
        output.set(mynetn.calc(X.get(), start_string.size()-1));
        output.add(mynetr.calc(X.get(), start_string.size()-1));
        output.div(output.sum());
        end_strings.push_back(index_to_char.at(get_weighted_random_index(output[0])));

        for(size_t i=start_string.size();i<start_string.size()+output_size-1;i++)
        {
            X.set(char_to_index.at(end_strings.back()));
            output.set(mynetn.calc(X.get(), i));
            output.add(mynetr.calc(X.get(), i));
            output.div(output.sum());
            end_strings.push_back(index_to_char.at(get_weighted_random_index(output[0])));
        }
    }

    {
        for(size_t i=0;i<start_string.size()-1;i++)
        {
            X.set(char_to_index.at(start_string[i]));
            mynetn.calc(X.get(), i);
            mynetr.calc(X.get(), i);
        }
        X.set(char_to_index.at(start_string.back()));
        Matrix<1,output_mem_size> output;
        output.set(mynetn.calc(X.get(), start_string.size()-1));
        output.mul(mynetr.calc(X.get(), start_string.size()-1));
        output.div(output.sum());
        end_stringm.push_back(index_to_char.at(get_weighted_random_index(output[0])));

        for(size_t i=start_string.size();i<start_string.size()+output_size-1;i++)
        {
            X.set(char_to_index.at(end_stringm.back()));
            output.set(mynetn.calc(X.get(), i));
            output.mul(mynetr.calc(X.get(), i));
            output.div(output.sum());
            end_stringm.push_back(index_to_char.at(get_weighted_random_index(output[0])));
        }
    }


    print(start_string+end_stringn);
    print(start_string+end_stringr);
    print(start_string+end_strings);
    print(start_string+end_stringm);


    return 0;
}