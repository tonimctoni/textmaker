// #include <iostream>
#include "matrix.hpp"
#include <array>
#include <random>

template<unsigned long input_size, unsigned long mem_cell_size, unsigned long output_size>
struct LstmParam
{
    static constexpr unsigned long concat_size=input_size+mem_cell_size;
    //LSTM weights
    Matrix<input_size, mem_cell_size> weights_xg;
    Matrix<input_size, mem_cell_size> weights_xi;
    Matrix<input_size, mem_cell_size> weights_xf;
    Matrix<input_size, mem_cell_size> weights_xo;
    Matrix<mem_cell_size, mem_cell_size> weights_hg;
    Matrix<mem_cell_size, mem_cell_size> weights_hi;
    Matrix<mem_cell_size, mem_cell_size> weights_hf;
    Matrix<mem_cell_size, mem_cell_size> weights_ho;
    //LSTM biases
    Matrix<1, mem_cell_size> bias_g;
    Matrix<1, mem_cell_size> bias_i;
    Matrix<1, mem_cell_size> bias_f;
    Matrix<1, mem_cell_size> bias_o;
    //LSTM memory cell to output weights and biases
    Matrix<mem_cell_size, output_size> weights_y;
    Matrix<1, output_size> bias_y;

    //Constructor: randomizes the weights and biases
    LstmParam()
    {
        weights_xg.randomize_for_nn(concat_size+1);
        weights_xi.randomize_for_nn(concat_size+1);
        weights_xf.randomize_for_nn(concat_size+1);
        weights_xo.randomize_for_nn(concat_size+1);
        weights_hg.randomize_for_nn(concat_size+1);
        weights_hi.randomize_for_nn(concat_size+1);
        weights_hf.randomize_for_nn(concat_size+1);
        weights_ho.randomize_for_nn(concat_size+1);
        bias_g.randomize_for_nn(concat_size+1);
        bias_i.randomize_for_nn(concat_size+1);
        bias_f.randomize_for_nn(concat_size+1);
        bias_o.randomize_for_nn(concat_size+1);
        weights_y.randomize_for_nn(mem_cell_size+1);
        bias_y.randomize_for_nn(mem_cell_size+1);
    }
};

template<unsigned long input_size, unsigned long mem_cell_size, unsigned long output_size>
struct LstmState
{
    static constexpr unsigned long concat_size=input_size+mem_cell_size;
    //LSTM states of inputs+h after passed through weights (synapses) and activation function applied to them.
    Matrix<1,mem_cell_size> state_g;
    Matrix<1,mem_cell_size> state_i;
    Matrix<1,mem_cell_size> state_f;
    Matrix<1,mem_cell_size> state_o;
    Matrix<1,mem_cell_size> state_s;
    //Further internal states
    Matrix<1,mem_cell_size> state_st;
    Matrix<1,mem_cell_size> state_h;
    Matrix<1,output_size> output;
    //Deltas
    Matrix<1,output_size> delta_output;
    Matrix<1,mem_cell_size> delta_h;
    Matrix<1,mem_cell_size> delta_o;
    Matrix<1,mem_cell_size> delta_s;
    Matrix<1,mem_cell_size> delta_i;
    Matrix<1,mem_cell_size> delta_g;
    Matrix<1,mem_cell_size> delta_f;
    //These two deltas are used by the next timestep (backpropagation through time)
    Matrix<1,mem_cell_size> delta_ls;//last s
    Matrix<1,mem_cell_size> delta_lh;//last h
};

//apply learning rate to first deltas (where i use "equals") ????
int do_stuff()
{
    static constexpr unsigned long input_size=2;
    static constexpr unsigned long mem_cell_size=6;
    static constexpr unsigned long output_size=1;
    static constexpr unsigned long time_steps=10;
    const double learning_rate=.2;

    std::array<LstmState<input_size,mem_cell_size,output_size>,time_steps> lstm_states;
    LstmParam<input_size,mem_cell_size,output_size> lstm_parameters;
    Matrix<time_steps,2> X;
    Matrix<time_steps,1> Y;

    std::random_device rd;
    std::mt19937 gen(rd());
    const size_t digits=time_steps-1;
    std::uniform_int_distribution<size_t> dst(0,(1<<digits)-1);
    for(size_t iteration=0;iteration<50000;iteration++)//10000000//5000000
    {
        {
            size_t x1=dst(gen);
            size_t x2=dst(gen);
            size_t y=x1+x2;
            for(size_t i=0;i<digits+1;i++)
            {
                X[i][0]=(x1>>i)&1;
                X[i][1]=(x2>>i)&1;
                Y[i][0]=(y>>i)&1;
            }
        }

        //Get output of LSTM+tanhunit given inputs X (1 row per timestep)
        for(size_t i=0;i<time_steps;i++)
        {
            //Calculate states g, i (input gate), f (forget gate), and o (output gate). steps are split for readability
            //f does not need to be calculated in the first round (nothing to forget there). maybe optimize that later.

            //Multiply input with corresponding weights for each state
            lstm_states[i].state_g.equals_row_of_a_dot_b(X,i,lstm_parameters.weights_xg);
            lstm_states[i].state_i.equals_row_of_a_dot_b(X,i,lstm_parameters.weights_xi);
            lstm_states[i].state_f.equals_row_of_a_dot_b(X,i,lstm_parameters.weights_xf);
            lstm_states[i].state_o.equals_row_of_a_dot_b(X,i,lstm_parameters.weights_xo);
            if(i!=0)
            {
                //Multiply last h-state with corresponding weights for each state
                lstm_states[i].state_g.add_a_dot_b(lstm_states[i-1].state_h, lstm_parameters.weights_hg);
                lstm_states[i].state_i.add_a_dot_b(lstm_states[i-1].state_h, lstm_parameters.weights_hi);
                lstm_states[i].state_f.add_a_dot_b(lstm_states[i-1].state_h, lstm_parameters.weights_hf);
                lstm_states[i].state_o.add_a_dot_b(lstm_states[i-1].state_h, lstm_parameters.weights_ho);
            }
            //Add biases to each state
            lstm_states[i].state_g.add(lstm_parameters.bias_g);
            lstm_states[i].state_i.add(lstm_parameters.bias_i);
            lstm_states[i].state_f.add(lstm_parameters.bias_f);
            lstm_states[i].state_o.add(lstm_parameters.bias_o);
            //Apply activation function to each state
            lstm_states[i].state_g.apply_tanh();
            lstm_states[i].state_i.apply_sigmoid();
            lstm_states[i].state_f.apply_sigmoid();
            lstm_states[i].state_o.apply_sigmoid();

            //Calculate s-state. This is the "memory state" which passes information to subsecuent timesteps
            if(i!=0)lstm_states[i].state_s.equals_a_mul_b_add_c_mul_d(lstm_states[i].state_g, lstm_states[i].state_i, lstm_states[i-1].state_s, lstm_states[i].state_f);
            else lstm_states[i].state_s.equals_a_mul_b(lstm_states[i].state_g, lstm_states[i].state_i);

            //The "memory state" s needs to have a element-wise tanh function applied to it for further calculations
            lstm_states[i].state_st.set(lstm_states[i].state_s);
            lstm_states[i].state_st.apply_tanh();

            //Calculate the output of the LSTM (tanh of output of mem-cell times output gate)
            lstm_states[i].state_h.equals_a_mul_b(lstm_states[i].state_st, lstm_states[i].state_o);

            //Calculate the actual output (outside of the LSTM unit)
            lstm_states[i].output.equals_a_dot_b(lstm_states[i].state_h, lstm_parameters.weights_y);
            lstm_states[i].output.add(lstm_parameters.bias_y);
            lstm_states[i].output.apply_tanh();
        }

        for(size_t i=time_steps-1;;)
        {
            //Get outputs delta
            lstm_states[i].delta_output.equals_row_of_a_sub_b(Y,i,lstm_states[i].output);
            lstm_states[i].delta_output.mult_after_func02(lstm_states[i].output);
            //Get delta of the h-state
            lstm_states[i].delta_h.equals_a_dot_bt(lstm_states[i].delta_output, lstm_parameters.weights_y);
            if(i!=time_steps-1)lstm_states[i].delta_h.add(lstm_states[i+1].delta_lh);
            //Get delta of the o-state
            lstm_states[i].delta_o.equals_a_mul_b(lstm_states[i].delta_h, lstm_states[i].state_st);
            lstm_states[i].delta_o.mult_after_func01(lstm_states[i].state_o);
            //Get delta of the s-state
            lstm_states[i].delta_s.equals_a_mul_b(lstm_states[i].delta_h, lstm_states[i].state_o);
            lstm_states[i].delta_s.mult_after_func02(lstm_states[i].state_st);
            if(i!=time_steps-1)lstm_states[i].delta_s.add(lstm_states[i+1].delta_ls);
            //add last rounds delivered delta !!! (except for last timestep)

            //Get delta of the i-state
            lstm_states[i].delta_i.equals_a_mul_b(lstm_states[i].delta_s, lstm_states[i].state_g);
            lstm_states[i].delta_i.mult_after_func01(lstm_states[i].state_i);
            //Get delta of the g-state
            lstm_states[i].delta_g.equals_a_mul_b(lstm_states[i].delta_s, lstm_states[i].state_i);
            lstm_states[i].delta_g.mult_after_func02(lstm_states[i].state_g);
            //Get delta of the f-state and last s-state //f state does not exist in the first round anyways
            //Both deltas are not needed in the first round, so they are not calculated
            if(i!=0)
            {
                lstm_states[i].delta_f.equals_a_mul_b(lstm_states[i].delta_s, lstm_states[i-1].state_s);
                lstm_states[i].delta_f.mult_after_func01(lstm_states[i].state_f);
                lstm_states[i].delta_ls.equals_a_mul_b(lstm_states[i].delta_s, lstm_states[i].state_f);

                lstm_states[i].delta_lh.equals_a_dot_bt(lstm_states[i].delta_i, lstm_parameters.weights_hi);
                lstm_states[i].delta_lh.add_a_dot_bt(lstm_states[i].delta_f, lstm_parameters.weights_hf);
                lstm_states[i].delta_lh.add_a_dot_bt(lstm_states[i].delta_o, lstm_parameters.weights_ho);
                lstm_states[i].delta_lh.add_a_dot_bt(lstm_states[i].delta_g, lstm_parameters.weights_hg);
            }
            if(i--==0)break;
        }

        //update weights for each layer
        for(size_t i=0;i<time_steps;i++)
        {
            lstm_states[i].delta_output.mul(learning_rate);
            lstm_states[i].delta_g.mul(learning_rate);
            lstm_states[i].delta_i.mul(learning_rate);
            lstm_states[i].delta_f.mul(learning_rate);
            lstm_states[i].delta_o.mul(learning_rate);

            lstm_parameters.weights_y.add_at_dot_b(lstm_states[i].state_h, lstm_states[i].delta_output);
            lstm_parameters.weights_xg.add_row_of_a_t_dot_b(X, i, lstm_states[i].delta_g);
            lstm_parameters.weights_xi.add_row_of_a_t_dot_b(X, i, lstm_states[i].delta_i);
            lstm_parameters.weights_xf.add_row_of_a_t_dot_b(X, i, lstm_states[i].delta_f);
            lstm_parameters.weights_xo.add_row_of_a_t_dot_b(X, i, lstm_states[i].delta_o);
            if(i!=0)
            {
                lstm_parameters.weights_hg.add_at_dot_b(lstm_states[i-1].state_h, lstm_states[i].delta_g);
                lstm_parameters.weights_hi.add_at_dot_b(lstm_states[i-1].state_h, lstm_states[i].delta_i);
                lstm_parameters.weights_hf.add_at_dot_b(lstm_states[i-1].state_h, lstm_states[i].delta_f);
                lstm_parameters.weights_ho.add_at_dot_b(lstm_states[i-1].state_h, lstm_states[i].delta_o);
            }
            lstm_parameters.bias_y.add(lstm_states[i].delta_output);
            lstm_parameters.bias_g.add(lstm_states[i].delta_g);
            lstm_parameters.bias_i.add(lstm_states[i].delta_i);
            lstm_parameters.bias_f.add(lstm_states[i].delta_f);
            lstm_parameters.bias_o.add(lstm_states[i].delta_o);

        }
    }

    int errors=0;
    for(size_t iteration=0;iteration<1000;iteration++)
    {
        {
            size_t x1=dst(gen);
            size_t x2=dst(gen);
            size_t y=x1+x2;
            for(size_t i=0;i<digits+1;i++)
            {
                X[i][0]=(x1>>i)&1;
                X[i][1]=(x2>>i)&1;
                Y[i][0]=(y>>i)&1;
            }
        }
        for(size_t i=0;i<time_steps;i++)
        {
            lstm_states[i].state_g.equals_row_of_a_dot_b(X,i,lstm_parameters.weights_xg);
            lstm_states[i].state_i.equals_row_of_a_dot_b(X,i,lstm_parameters.weights_xi);
            lstm_states[i].state_f.equals_row_of_a_dot_b(X,i,lstm_parameters.weights_xf);
            lstm_states[i].state_o.equals_row_of_a_dot_b(X,i,lstm_parameters.weights_xo);
            if(i!=0)
            {
                lstm_states[i].state_g.add_a_dot_b(lstm_states[i-1].state_h, lstm_parameters.weights_hg);
                lstm_states[i].state_i.add_a_dot_b(lstm_states[i-1].state_h, lstm_parameters.weights_hi);
                lstm_states[i].state_f.add_a_dot_b(lstm_states[i-1].state_h, lstm_parameters.weights_hf);
                lstm_states[i].state_o.add_a_dot_b(lstm_states[i-1].state_h, lstm_parameters.weights_ho);
            }
            lstm_states[i].state_g.add(lstm_parameters.bias_g);
            lstm_states[i].state_i.add(lstm_parameters.bias_i);
            lstm_states[i].state_f.add(lstm_parameters.bias_f);
            lstm_states[i].state_o.add(lstm_parameters.bias_o);
            lstm_states[i].state_g.apply_tanh();
            lstm_states[i].state_i.apply_sigmoid();
            lstm_states[i].state_f.apply_sigmoid();
            lstm_states[i].state_o.apply_sigmoid();
            if(i!=0)lstm_states[i].state_s.equals_a_mul_b_add_c_mul_d(lstm_states[i].state_g, lstm_states[i].state_i, lstm_states[i-1].state_s, lstm_states[i].state_f);
            else lstm_states[i].state_s.equals_a_mul_b(lstm_states[i].state_g, lstm_states[i].state_i);
            lstm_states[i].state_st.set(lstm_states[i].state_s);
            lstm_states[i].state_st.apply_tanh();
            lstm_states[i].state_h.equals_a_mul_b(lstm_states[i].state_st, lstm_states[i].state_o);
            lstm_states[i].output.equals_a_dot_b(lstm_states[i].state_h, lstm_parameters.weights_y);
            lstm_states[i].output.add(lstm_parameters.bias_y);
            lstm_states[i].output.apply_tanh();
        }

        // for(size_t i=0;i<time_steps;i++) std::cout<<X[i][0]<<" ";std::cout<<std::endl;
        // for(size_t i=0;i<time_steps;i++) std::cout<<X[i][1]<<" ";std::cout<<std::endl;
        // for(size_t i=0;i<time_steps;i++) std::cout<<"--";std::cout<<std::endl;
        // for(size_t i=0;i<time_steps;i++) std::cout<<(lstm_states[i].output[0][0]>.5?1:0)<<" ";std::cout<<std::endl;
        // std::cout<<std::endl;

        for(size_t i=0;i<time_steps;i++)
        {
            if((lstm_states[i].output[0][0]>.5?1.0:0.0)!=Y[i][0])
            {
                errors++;
                break;
            }
        }


        // print(X[0][0]>.5?1:0, X[1][0]>.5?1:0, X[2][0]>.5?1:0, X[3][0]>.5?1:0, X[4][0]>.5?1:0, X[5][0]>.5?1:0, X[6][0]>.5?1:0, X[7][0]>.5?1:0);
        // print(X[0][1]>.5?1:0, X[1][1]>.5?1:0, X[2][1]>.5?1:0, X[3][1]>.5?1:0, X[4][1]>.5?1:0, X[5][1]>.5?1:0, X[6][1]>.5?1:0, X[7][1]>.5?1:0);
        // print("----------------");
        // print(lstm_states[0].output[0][0]>.5?1:0, lstm_states[1].output[0][0]>.5?1:0, lstm_states[2].output[0][0]>.5?1:0, lstm_states[3].output[0][0]>.5?1:0,
        //     lstm_states[4].output[0][0]>.5?1:0, lstm_states[5].output[0][0]>.5?1:0, lstm_states[6].output[0][0]>.5?1:0, lstm_states[7].output[0][0]>.5?1:0);
        // print();
    }
    // print(errors);
    return errors;
}

int main()
{
    int not_succ=0;
    for(int i=1;;i++)
    {
        if(do_stuff()!=0) not_succ++;
        print(i, not_succ);
    }

    return 0;
}