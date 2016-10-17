#include <iostream>
#include "matrix.hpp"
#include <array>
#include <random>



//use tanh after
int main()
{
    const std::array<Matrix<8,1>,8> Xs{
        Matrix<8,1>{{0},{0},{0},{0},{0},{0},{0},{1}},
        Matrix<8,1>{{0},{0},{0},{0},{0},{0},{1},{0}},
        Matrix<8,1>{{0},{0},{0},{0},{0},{1},{0},{0}},
        Matrix<8,1>{{0},{0},{0},{0},{1},{0},{0},{0}},
        Matrix<8,1>{{0},{0},{0},{1},{0},{0},{0},{0}},
        Matrix<8,1>{{0},{0},{1},{0},{0},{0},{0},{0}},
        Matrix<8,1>{{0},{1},{0},{0},{0},{0},{0},{0}},
        Matrix<8,1>{{1},{0},{0},{0},{0},{0},{0},{0}},
    };

    const std::array<Matrix<8,1>,8> Ys{
        Matrix<8,1>{{0},{0},{0},{0},{0},{0},{0},{1}},
        Matrix<8,1>{{0},{0},{0},{0},{0},{0},{1},{1}},
        Matrix<8,1>{{0},{0},{0},{0},{0},{1},{1},{1}},
        Matrix<8,1>{{0},{0},{0},{0},{1},{1},{1},{1}},
        Matrix<8,1>{{0},{0},{0},{1},{1},{1},{1},{1}},
        Matrix<8,1>{{0},{0},{1},{1},{1},{1},{1},{1}},
        Matrix<8,1>{{0},{1},{1},{1},{1},{1},{1},{1}},
        Matrix<8,1>{{1},{1},{1},{1},{1},{1},{1},{1}},
    };

    // const std::array<Matrix<8,1>,8> Ys{
    //     Matrix<8,1>{{0},{0},{0},{0},{0},{0},{0},{1}},
    //     Matrix<8,1>{{0},{0},{0},{0},{0},{0},{1},{0}},
    //     Matrix<8,1>{{0},{0},{0},{0},{0},{1},{0},{0}},
    //     Matrix<8,1>{{0},{0},{0},{0},{1},{0},{0},{0}},
    //     Matrix<8,1>{{0},{0},{0},{1},{0},{0},{0},{0}},
    //     Matrix<8,1>{{0},{0},{1},{0},{0},{0},{0},{0}},
    //     Matrix<8,1>{{0},{1},{0},{0},{0},{0},{0},{0}},
    //     Matrix<8,1>{{1},{0},{0},{0},{0},{0},{0},{0}},
    // };

    const double learning_rate=.8;
    Matrix<1,2> syn0;
    Matrix<2,2> synh;
    Matrix<2,1> syn1;
    Matrix<1,2> b1;
    Matrix<1,1> b2;
    syn0.randomize_for_nn();
    synh.randomize_for_nn();
    syn1.randomize_for_nn();
    b1.randomize_for_nn(2);
    b2.randomize_for_nn(1);

    std::array<Matrix<1,2>,8> l1s;
    std::array<Matrix<1,1>,8> l2s;
    std::array<Matrix<1,2>,8> l1_deltas;
    std::array<Matrix<1,1>,8> l2_deltas;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dst(0,7);
    for(size_t iteration=0;iteration<500000;iteration++)//10000000//5000000
    {
        size_t current_xy=dst(gen);

        //calculate the outputs of each layer
        for(size_t i=0;i<Xs[current_xy].size();i++)
        {
            l1s[i].equals_row_of_a_dot_b(Xs[current_xy],i,syn0);
            if(i!=0)l1s[i].add_a_dot_b(l1s[i-1], synh);
            l1s[i].add(b1);
            l1s[i].apply_sigmoid();
            l2s[i].equals_a_dot_b(l1s[i], syn1);
            l2s[i].add(b2);
            l2s[i].apply_sigmoid();
        }

        //calculate the deltas for each layer
        for(size_t i=Xs[current_xy].size()-1;;)
        {
            l2_deltas[i].equals_row_of_a_sub_b(Ys[current_xy],i,l2s[i]);
            l2_deltas[i].mult_after_func01(l2s[i]);
            l1_deltas[i].equals_a_dot_bt(l2_deltas[i], syn1);
            if(i!=Xs[current_xy].size()-1) l1_deltas[i].add_a_dot_bt(l1_deltas[i+1], synh);
            l1_deltas[i].mult_after_func01(l1s[i]);
            if(i--==0)break;
        }

        //update weights for each layer
        for(size_t i=0;i<Xs[current_xy].size();i++)
        {
            l1_deltas[i].mul(learning_rate);
            l2_deltas[i].mul(learning_rate);
            syn1.add_at_dot_b(l1s[i], l2_deltas[i]);
            if(i!=0) synh.add_at_dot_b(l1s[i-1], l1_deltas[i]);
            syn0.add_row_of_a_t_dot_b(Xs[current_xy], i, l1_deltas[i]);
            b2.add(l2_deltas[i]);
            b1.add(l1_deltas[i]);
        }
    }

    for(size_t cxy=0;cxy<8;cxy++)
    {
        print(Xs[cxy][0][0]>.5?1:0, Xs[cxy][1][0]>.5?1:0, Xs[cxy][2][0]>.5?1:0, Xs[cxy][3][0]>.5?1:0, Xs[cxy][4][0]>.5?1:0, Xs[cxy][5][0]>.5?1:0, Xs[cxy][6][0]>.5?1:0, Xs[cxy][7][0]>.5?1:0);
        for(size_t i=0;i<Xs[cxy].size();i++)
        {
            l1s[i].equals_row_of_a_dot_b(Xs[cxy],i,syn0);
            if(i!=0)l1s[i].add_a_dot_b(l1s[i-1], synh);
            l1s[i].add(b1);
            l1s[i].apply_sigmoid();
            l2s[i].equals_a_dot_b(l1s[i], syn1);
            l2s[i].add(b2);
            l2s[i].apply_sigmoid();
        }
        print(l2s[0][0][0]>.5?1:0, l2s[1][0][0]>.5?1:0, l2s[2][0][0]>.5?1:0, l2s[3][0][0]>.5?1:0, l2s[4][0][0]>.5?1:0, l2s[5][0][0]>.5?1:0, l2s[6][0][0]>.5?1:0, l2s[7][0][0]>.5?1:0);
        print();

    }


    print(syn0);
    print(syn1);
    print(b1);
    print(b2);

    return 0;
}