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

class MyVNet
{
private:
    static constexpr size_t allowed_char_amount=46;
    static constexpr unsigned long input_size=allowed_char_amount;
    static constexpr unsigned long reduced_input_size=allowed_char_amount/4;
    static constexpr unsigned long first_mem_cell_size=400;
    static constexpr unsigned long second_mem_cell_size=200;
    static constexpr unsigned long third_mem_cell_size=100;
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


int main()
{
    static constexpr size_t output_size=2000;
    // MyNet myneta("outs/26a.wab");
    // MyNet mynetn("outs/26n.wab");
    // MyNet mynetr("outs/26r.wab");
    // MyNet mynets("outs/26s.wab");
    MyVNet myvneta("outs/56va.wab");
    MyVNet myvnetn("outs/56vn.wab");
    MyVNet myvnetr("outs/56vr.wab");
    MyVNet myvnets("outs/56vs.wab");
    // myneta.set_time_steps(output_size);
    // mynetn.set_time_steps(output_size);
    // mynetr.set_time_steps(output_size);
    // mynets.set_time_steps(output_size);
    myvneta.set_time_steps(output_size);
    myvnetn.set_time_steps(output_size);
    myvnetr.set_time_steps(output_size);
    myvnets.set_time_steps(output_size);

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
        // const Matrix<1,output_mem_size>& outsa=myneta.calc(X.get(), i);
        // const Matrix<1,output_mem_size>& outsn=mynetn.calc(X.get(), i);
        // const Matrix<1,output_mem_size>& outsr=mynetr.calc(X.get(), i);
        // const Matrix<1,output_mem_size>& outss=mynets.calc(X.get(), i);
        const Matrix<1,output_mem_size>& outva=myvneta.calc(X.get(), i);
        const Matrix<1,output_mem_size>& outvn=myvnetn.calc(X.get(), i);
        const Matrix<1,output_mem_size>& outvr=myvnetr.calc(X.get(), i);
        const Matrix<1,output_mem_size>& outvs=myvnets.calc(X.get(), i);
        Matrix<1,output_mem_size> output(1.0);
        // output.mul(outsa);
        // output.mul(outsn);
        // output.mul(outsr);
        // output.mul(outss);
        output.mul(outva);
        output.mul(outvn);
        output.mul(outvr);
        output.mul(outvs);
        output[0][char_to_index['.']]*=10;
        output.div(output.sum());
        // print(output);
        size_t new_char_index=get_weighted_random_index(output[0]);
        cout << index_to_char[new_char_index];
        if(new_char_index==char_to_index['.']) break;
        X.set(new_char_index);
    }
    print();
    return 0;
}