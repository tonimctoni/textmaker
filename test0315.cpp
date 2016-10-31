#include "perceptron_timeseries_class.hpp"
#include "LSTM_class.hpp"
#include "softmax_timeseries_class.hpp"
#include <unordered_map>
// #include <ctime>
#include <memory>
using namespace std;

class MyNet
{
private:
    static constexpr size_t allowed_char_amount=46;
    static constexpr unsigned long input_size=allowed_char_amount;
    static constexpr unsigned long reduced_input_size=allowed_char_amount/4;
    static constexpr unsigned long first_mem_cell_size=200;
    static constexpr unsigned long second_mem_cell_size=100;
    static constexpr unsigned long output_mem_size=allowed_char_amount;

    using Block01=BaseTahnPerceptronBlock<input_size,reduced_input_size>;
    using Block02=BaseLSTMBlock<reduced_input_size, first_mem_cell_size>;
    using Block03=BaseLSTMBlock<first_mem_cell_size, second_mem_cell_size>;
    using Block05=BaseSoftmaxBlock<second_mem_cell_size, output_mem_size>;

    unique_ptr<Block01> perceptronblock;
    unique_ptr<Block02> lstmblock1;
    unique_ptr<Block03> lstmblock2;
    unique_ptr<Block05> softmaxblock;
public:
    MyNet(const char *filename)noexcept: perceptronblock(new Block01), lstmblock1(new Block02), lstmblock2(new Block03), softmaxblock(new Block05)
    {
        ifstream in(filename);
        assert(in.good());
        perceptronblock->only_wb_from_bin_file(in);
        lstmblock1->only_wb_from_bin_file(in);
        lstmblock2->only_wb_from_bin_file(in);
        softmaxblock->only_wb_from_bin_file(in);
    }

    inline void set_time_steps(size_t time_steps) noexcept
    {
        perceptronblock->set_time_steps(time_steps);
        lstmblock1->set_time_steps(time_steps);
        lstmblock2->set_time_steps(time_steps);
        softmaxblock->set_time_steps(time_steps);
    }

    inline const Matrix<1,output_mem_size>& calc(const Matrix<1,input_size> &X, size_t time_step) noexcept
    {
        perceptronblock->calc(X, time_step);
        lstmblock1->calc(perceptronblock->get_output(time_step), time_step);
        lstmblock2->calc(lstmblock1->get_output(time_step), time_step);
        softmaxblock->calc(lstmblock2->get_output(time_step), time_step);
        return softmaxblock->get_output(time_step);
    }
};

template<unsigned long first_mem_cell_size, unsigned long second_mem_cell_size, unsigned long third_mem_cell_size>
class MyVNet
{
private:
    static constexpr size_t allowed_char_amount=46;
    static constexpr unsigned long input_size=allowed_char_amount;
    static constexpr unsigned long reduced_input_size=allowed_char_amount/4;
    // static constexpr unsigned long first_mem_cell_size=400;
    // static constexpr unsigned long second_mem_cell_size=200;
    // static constexpr unsigned long third_mem_cell_size=100;
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
    MyVNet(const char *filename)noexcept: perceptronblock(new Block01), lstmblock1(new Block02), lstmblock2(new Block03), lstmblock3(new Block04), softmaxblock(new Block05)
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

// output[0][char_to_index['.']]*=10;
int main()
{
    static constexpr size_t output_size=2000;
    MyVNet<512,256,128> mynetn("outs/tn.wab");
    MyVNet<512,256,128> mynetr("outs/tr.wab");
    mynetn.set_time_steps(output_size);
    mynetr.set_time_steps(output_size);

    //Setup the char_to_index and index_to_char mappings
    static constexpr size_t allowed_char_amount=46;
    const string index_to_char="! ')(-,.103254769;:?acbedgfihkjmlonqpsrutwvyxz";
    unordered_map<char, size_t> char_to_index;for(size_t i=0;i<index_to_char.size();i++) char_to_index[index_to_char[i]]=i;
    assert(index_to_char.size()==allowed_char_amount and char_to_index.size()==allowed_char_amount);

    static constexpr unsigned long input_size=allowed_char_amount;
    static constexpr unsigned long output_mem_size=allowed_char_amount;
    OneHot<input_size> X;
    X.set(char_to_index['.']);
    for(size_t i=0;i<output_size;i++)
    {
        const Matrix<1,output_mem_size>& outn=mynetn.calc(X.get(), i);
        const Matrix<1,output_mem_size>& outr=mynetr.calc(X.get(), i);
        // Matrix<1,output_mem_size> output(1.0);
        // output.mul(outn);
        // output.mul(outr);
        Matrix<1,output_mem_size> output;
        {
            output.set(0.0);
            output.add(outn);
            output.add(outr);
            for(size_t j=0;j<output_mem_size;j++) if(outn[0][j]<.01 or outr[0][j]<.01) output[0][j]=.0;
            if(output.sum()==0)
            {
                output.set(outn);
                output.mul(outr);
                cout << "(**)";
            }
        }
        output.div(output.sum());
        size_t new_char_index=get_weighted_random_index(output[0]);
        cout << index_to_char[new_char_index];
        // if(new_char_index==char_to_index['.']) break;
        X.set(new_char_index);
    }
    print();
    return 0;
}