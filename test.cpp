#include <iostream>
#include "matrix.hpp"





int main()
{
    Matrix<4,2> X{{0,0},{0,1},{1,0},{1,1}};
    Matrix<4,1> Y{{0},{1},{1},{0}};
    const double learning_rate=.8;
    Matrix<2,3> syn0;
    Matrix<3,1> syn1;
    Matrix<1,3> b1;
    Matrix<1,1> b2;
    syn0.randomize_for_nn();
    syn1.randomize_for_nn();
    b1.randomize_for_nn(3);9.396
    b2.randomize_for_nn(1);
    Matrix<1,3> l1;
    Matrix<1,1> l2;
    Matrix<1,3> l1_delta;
    Matrix<1,1> l2_delta;
    static_assert(X.size()==Y.size(),"len(X)!=len(Y)");
    //10000000 --> 9.396
    for(size_t i=0;i<10000000;i++)//10000000
    {
        for(size_t j=0;j<X.size();j++)
        {
            l1.equals_row_of_a_dot_b(X,j,syn0);
            l1.add(b1);
            l1.apply_sigmoid();
            l2.equals_a_dot_b(l1, syn1);
            l2.add(b2);
            l2.apply_sigmoid();

            l2_delta.equals_row_of_a_sub_b(Y,j,l2);
            l2_delta.mult_after_func01(l2);
            l1_delta.equals_a_dot_bt(l2_delta, syn1);
            l1_delta.mult_after_func01(l1);

            l1_delta.mul(learning_rate);
            l2_delta.mul(learning_rate);
            syn1.add_at_dot_b(l1, l2_delta);
            syn0.add_row_of_a_t_dot_b(X, j, l1_delta);
            b2.add(l2_delta);
            b1.add(l1_delta);
        }
    }

    for(size_t i=0;i<X.size();i++)
    {
        l1.equals_row_of_a_dot_b(X,i,syn0);
        l1.add(b1);
        l1.apply_sigmoid();
        l2.equals_a_dot_b(l1, syn1);
        l2.add(b2);
        l2.apply_sigmoid();
        print(l2);
    }

    return 0;
}