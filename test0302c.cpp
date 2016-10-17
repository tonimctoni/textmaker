#include "lstm_block_class.hpp"
#include <memory>
using namespace std;



int main()
{
    static constexpr unsigned long input_size=2;
    static constexpr unsigned long mem_cell_size=6;
    static constexpr unsigned long output_size=1;
    const double learning_rate=.2;

    const size_t digits=8;
    const size_t time_steps=digits;
    vector<Matrix<1,2>> X(time_steps);
    vector<Matrix<1,1>> Y(time_steps);

    LSTMBlock<input_size, mem_cell_size> lstmblock(time_steps);
    TahnPerceptronBlock<mem_cell_size, output_size> precblock(time_steps);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dst(0,(1<<(digits-1))-1);
    for(size_t iteration=0;iteration<50000;iteration++)
    {
        {
            size_t x1=dst(gen);
            size_t x2=dst(gen);
            size_t y=x1+x2;
            for(size_t i=0;i<digits;i++)
            {
                X[i][0][0]=(x1>>i)&1;
                X[i][0][1]=(x2>>i)&1;
                Y[i][0][0]=(y>>i)&1;
            }
        }

        for(size_t i=0;i<time_steps;i++)
        {
            lstmblock.calc(X[i], i);
            precblock.calc(lstmblock.get_output(i), i);
        }
        for(size_t i=time_steps-1;;)
        {
            precblock.set_first_delta(Y[i], i);
            precblock.propagate_delta(lstmblock.get_delta_output(i), i);
            lstmblock.propagate_delta(i, time_steps);
            if(i--==0)break;
        }
        for(size_t i=0;i<time_steps;i++)
        {
            lstmblock.update_weights(X[i], i, learning_rate);
            precblock.update_weights(lstmblock.get_output(i), i, learning_rate);
        }
    }


    int errors=0;
    for(size_t iteration=0;iteration<10000;iteration++)
    {
        {
            size_t x1=dst(gen);
            size_t x2=dst(gen);
            size_t y=x1+x2;
            for(size_t i=0;i<digits;i++)
            {
                X[i][0][0]=(x1>>i)&1;
                X[i][0][1]=(x2>>i)&1;
                Y[i][0][0]=(y>>i)&1;
            }
        }

        for(size_t i=0;i<time_steps;i++)
        {
            lstmblock.calc(X[i], i);
            precblock.calc(lstmblock.get_output(i), i);;
        }

        // for(size_t i=0;i<time_steps;i++) std::cout<<X[i][0][0]<<" ";std::cout<<std::endl;
        // for(size_t i=0;i<time_steps;i++) std::cout<<X[i][0][1]<<" ";std::cout<<std::endl;
        // for(size_t i=0;i<time_steps;i++) std::cout<<"--";std::cout<<std::endl;
        // for(size_t i=0;i<time_steps;i++) std::cout<<(layer_states[i].output[0][0]>.5?1:0)<<" ";std::cout<<std::endl;
        // std::cout<<std::endl;

        for(size_t i=0;i<time_steps;i++)
        {
            if((precblock.get_output(i)[0][0]>.5?1.0:0.0)!=Y[i][0][0])
            {
                errors++;
                break;
            }
        }
    }

    print("There have been", errors, "errors.");
    return 0;
}