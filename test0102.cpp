#include "lstm_block_class.hpp"
#include <memory>
using namespace std;


int main()
{
    unique_ptr<TahnPerceptronBlock<2,3>> block1(new TahnPerceptronBlock<2,3>(1));
    unique_ptr<TahnPerceptronBlock<3,1>> block2(new TahnPerceptronBlock<3,1>(2));

    vector<Matrix<1,2>> X{{{0,0},},{{1,0},},{{0,1},},{{1,1},},};
    vector<Matrix<1,1>> Y{{{0, },},{{1, },},{{1, },},{{0, },},};
    print("################################");
    for(size_t i=0;i<10000000;i++)//10000000 -> 0m13.372s
    {
        for(size_t j=0;j<4;j++)
        {
            block1->calc(X[j], 0);
            block2->calc(block1->get_output(0), 0);

            block2->set_first_delta(Y[j], 0);
            block2->propagate_delta(block1->get_delta_output(0), 0);
            block1->propagate_delta(0);

            block1->update_weights(X[j],0,0.2);
            block2->update_weights(block1->get_output(0),0,0.2);
        }
    }

    for(size_t j=0;j<4;j++)
    {
        block1->calc(X[j], 0);
        block2->calc(block1->get_output(0), 0);

        print(block2->get_output(0));
    }
    return 0;
}