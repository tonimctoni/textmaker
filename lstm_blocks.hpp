#include "matrix.hpp"
#include <vector>

template<unsigned long input_size, unsigned long mem_cell_size>
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
    // Matrix<mem_cell_size, output_size> weights_y;
    // Matrix<1, output_size> bias_y;

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
        // weights_y.randomize_for_nn(mem_cell_size+1);
        // bias_y.randomize_for_nn(mem_cell_size+1);
    }

    inline void to_file(std::ofstream &out)
    {
        weights_xg.to_file(out);
        weights_xi.to_file(out);
        weights_xf.to_file(out);
        weights_xo.to_file(out);
        weights_hg.to_file(out);
        weights_hi.to_file(out);
        weights_hf.to_file(out);
        weights_ho.to_file(out);
        bias_g.to_file(out);
        bias_i.to_file(out);
        bias_f.to_file(out);
        bias_o.to_file(out);
    }

    inline void from_file(std::ifstream &in)
    {
        weights_xg.from_file(in);
        weights_xi.from_file(in);
        weights_xf.from_file(in);
        weights_xo.from_file(in);
        weights_hg.from_file(in);
        weights_hi.from_file(in);
        weights_hf.from_file(in);
        weights_ho.from_file(in);
        bias_g.from_file(in);
        bias_i.from_file(in);
        bias_f.from_file(in);
        bias_o.from_file(in);
    }
};


template<unsigned long input_size, unsigned long mem_cell_size>
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
    // Matrix<1,output_size> output;
    //Deltas
    // Matrix<1,output_size> delta_output;
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

template<unsigned long input_size, unsigned long output_size>
struct LayerParam
{
    Matrix<input_size, output_size> weights;
    Matrix<1, output_size> bias;

    LayerParam()
    {
        weights.randomize_for_nn(input_size+1);
        bias.randomize_for_nn(input_size+1);
    }

    inline void to_file(std::ofstream &out)
    {
        weights.to_file(out);
        bias.to_file(out);
    }

    inline void from_file(std::ifstream &in)
    {
        weights.from_file(in);
        bias.from_file(in);
    }
};

template<unsigned long input_size, unsigned long output_size>
struct LayerState
{
    Matrix<1,output_size> output;
    Matrix<1,output_size> delta_output;
};



/*
     _      _____ ________  ___
    | |    /  ___|_   _|  \/  |
    | |    \ `--.  | | | .  . |
    | |     `--. \ | | | |\/| |
    | |____/\__/ / | | | |  | |
    \_____/\____/  \_/ \_|  |_/
*/

template<unsigned long input_size, unsigned long mem_cell_size>
inline void calc_lstm(const Matrix<1,input_size> &X, std::vector<LstmState<input_size,mem_cell_size>> &lstm_states, const LstmParam<input_size,mem_cell_size> &lstm_parameters, size_t time_step)
{
    assert(time_step<lstm_states.size());
    //Calculate states g, i (input gate), f (forget gate), and o (output gate). steps are split for readability
    //f does not need to be calculated in the first round (nothing to forget there). maybe optimize that later.

    //Multiply input with corresponding weights for each state
    lstm_states[time_step].state_g.equals_a_dot_b(X,lstm_parameters.weights_xg);
    lstm_states[time_step].state_i.equals_a_dot_b(X,lstm_parameters.weights_xi);
    lstm_states[time_step].state_f.equals_a_dot_b(X,lstm_parameters.weights_xf);
    lstm_states[time_step].state_o.equals_a_dot_b(X,lstm_parameters.weights_xo);
    if(time_step!=0)
    {
        //Multiply last h-state with corresponding weights for each state
        lstm_states[time_step].state_g.add_a_dot_b(lstm_states[time_step-1].state_h, lstm_parameters.weights_hg);
        lstm_states[time_step].state_i.add_a_dot_b(lstm_states[time_step-1].state_h, lstm_parameters.weights_hi);
        lstm_states[time_step].state_f.add_a_dot_b(lstm_states[time_step-1].state_h, lstm_parameters.weights_hf);
        lstm_states[time_step].state_o.add_a_dot_b(lstm_states[time_step-1].state_h, lstm_parameters.weights_ho);
    }
    //Add biases to each state
    lstm_states[time_step].state_g.add(lstm_parameters.bias_g);
    lstm_states[time_step].state_i.add(lstm_parameters.bias_i);
    lstm_states[time_step].state_f.add(lstm_parameters.bias_f);
    lstm_states[time_step].state_o.add(lstm_parameters.bias_o);
    //Apply activation function to each state
    lstm_states[time_step].state_g.apply_tanh();
    lstm_states[time_step].state_i.apply_sigmoid();
    lstm_states[time_step].state_f.apply_sigmoid();
    lstm_states[time_step].state_o.apply_sigmoid();

    //Calculate s-state. This is the "memory state" which passes information to subsecuent timesteps
    if(time_step!=0)lstm_states[time_step].state_s.equals_a_mul_b_add_c_mul_d(lstm_states[time_step].state_g, lstm_states[time_step].state_i, lstm_states[time_step-1].state_s, lstm_states[time_step].state_f);
    else lstm_states[time_step].state_s.equals_a_mul_b(lstm_states[time_step].state_g, lstm_states[time_step].state_i);

    //The "memory state" s needs to have a element-wise tanh function applied to it for further calculations
    lstm_states[time_step].state_st.set(lstm_states[time_step].state_s);
    lstm_states[time_step].state_st.apply_tanh();

    //Calculate the output of the LSTM (tanh of output of mem-cell times output gate)
    lstm_states[time_step].state_h.equals_a_mul_b(lstm_states[time_step].state_st, lstm_states[time_step].state_o);
}

template<unsigned long input_size, unsigned long mem_cell_size>
inline void set_first_delta_lstm(const Matrix<1,mem_cell_size> &Y, std::vector<LstmState<input_size,mem_cell_size>> &lstm_states, size_t time_step)
{
    assert(time_step<lstm_states.size());
    //Get outputs delta
    lstm_states[time_step].delta_h.equals_a_sub_b(Y,lstm_states[time_step].state_h);
}

template<unsigned long input_size, unsigned long mem_cell_size>
inline void propagate_delta_lstm(std::vector<LstmState<input_size,mem_cell_size>> &lstm_states, const LstmParam<input_size,mem_cell_size> &lstm_parameters, size_t time_step, size_t total_time_steps)
{
    assert(time_step<lstm_states.size() && time_step<total_time_steps);
    //Add deltas from future timesteps to the h-state delta
    if(time_step<total_time_steps-1)lstm_states[time_step].delta_h.add(lstm_states[time_step+1].delta_lh);
    //Get delta of the o-state
    lstm_states[time_step].delta_o.equals_a_mul_b(lstm_states[time_step].delta_h, lstm_states[time_step].state_st);
    lstm_states[time_step].delta_o.mult_after_func01(lstm_states[time_step].state_o);
    //Get delta of the s-state
    lstm_states[time_step].delta_s.equals_a_mul_b(lstm_states[time_step].delta_h, lstm_states[time_step].state_o);
    lstm_states[time_step].delta_s.mult_after_func02(lstm_states[time_step].state_st);
    if(time_step!=total_time_steps-1)lstm_states[time_step].delta_s.add(lstm_states[time_step+1].delta_ls);

    //Get delta of the i-state
    lstm_states[time_step].delta_i.equals_a_mul_b(lstm_states[time_step].delta_s, lstm_states[time_step].state_g);
    lstm_states[time_step].delta_i.mult_after_func01(lstm_states[time_step].state_i);
    //Get delta of the g-state
    lstm_states[time_step].delta_g.equals_a_mul_b(lstm_states[time_step].delta_s, lstm_states[time_step].state_i);
    lstm_states[time_step].delta_g.mult_after_func02(lstm_states[time_step].state_g);
    //Get delta of the f-state and last s-state //f state does not exist in the first round anyways
    //Both deltas are not needed in the first round, so they are not calculated
    if(time_step!=0)
    {
        lstm_states[time_step].delta_f.equals_a_mul_b(lstm_states[time_step].delta_s, lstm_states[time_step-1].state_s);
        lstm_states[time_step].delta_f.mult_after_func01(lstm_states[time_step].state_f);
        lstm_states[time_step].delta_ls.equals_a_mul_b(lstm_states[time_step].delta_s, lstm_states[time_step].state_f);

        lstm_states[time_step].delta_lh.equals_a_dot_bt(lstm_states[time_step].delta_i, lstm_parameters.weights_hi);
        lstm_states[time_step].delta_lh.add_a_dot_bt(lstm_states[time_step].delta_f, lstm_parameters.weights_hf);
        lstm_states[time_step].delta_lh.add_a_dot_bt(lstm_states[time_step].delta_o, lstm_parameters.weights_ho);
        lstm_states[time_step].delta_lh.add_a_dot_bt(lstm_states[time_step].delta_g, lstm_parameters.weights_hg);
    }
}

template<unsigned long input_size, unsigned long mem_cell_size>
inline void propagate_delta_lstm(Matrix<1,input_size> &X_delta, std::vector<LstmState<input_size,mem_cell_size>> &lstm_states, const LstmParam<input_size,mem_cell_size> &lstm_parameters, size_t time_step, size_t total_time_steps)
{
    assert(time_step<lstm_states.size() && time_step<total_time_steps);
    //Add deltas from future timesteps to the h-state delta
    if(time_step<total_time_steps-1)lstm_states[time_step].delta_h.add(lstm_states[time_step+1].delta_lh);
    //Get delta of the o-state
    lstm_states[time_step].delta_o.equals_a_mul_b(lstm_states[time_step].delta_h, lstm_states[time_step].state_st);
    lstm_states[time_step].delta_o.mult_after_func01(lstm_states[time_step].state_o);
    //Get delta of the s-state
    lstm_states[time_step].delta_s.equals_a_mul_b(lstm_states[time_step].delta_h, lstm_states[time_step].state_o);
    lstm_states[time_step].delta_s.mult_after_func02(lstm_states[time_step].state_st);
    if(time_step!=total_time_steps-1)lstm_states[time_step].delta_s.add(lstm_states[time_step+1].delta_ls);

    //Get delta of the i-state
    lstm_states[time_step].delta_i.equals_a_mul_b(lstm_states[time_step].delta_s, lstm_states[time_step].state_g);
    lstm_states[time_step].delta_i.mult_after_func01(lstm_states[time_step].state_i);
    //Get delta of the g-state
    lstm_states[time_step].delta_g.equals_a_mul_b(lstm_states[time_step].delta_s, lstm_states[time_step].state_i);
    lstm_states[time_step].delta_g.mult_after_func02(lstm_states[time_step].state_g);
    //Get delta of the f-state and last s-state //f state does not exist in the first round anyways
    //Both deltas are not needed in the first round, so they are not calculated
    if(time_step!=0)
    {
        lstm_states[time_step].delta_f.equals_a_mul_b(lstm_states[time_step].delta_s, lstm_states[time_step-1].state_s);
        lstm_states[time_step].delta_f.mult_after_func01(lstm_states[time_step].state_f);
        lstm_states[time_step].delta_ls.equals_a_mul_b(lstm_states[time_step].delta_s, lstm_states[time_step].state_f);

        lstm_states[time_step].delta_lh.equals_a_dot_bt(lstm_states[time_step].delta_i, lstm_parameters.weights_hi);
        lstm_states[time_step].delta_lh.add_a_dot_bt(lstm_states[time_step].delta_f, lstm_parameters.weights_hf);
        lstm_states[time_step].delta_lh.add_a_dot_bt(lstm_states[time_step].delta_o, lstm_parameters.weights_ho);
        lstm_states[time_step].delta_lh.add_a_dot_bt(lstm_states[time_step].delta_g, lstm_parameters.weights_hg);
    }

    X_delta.equals_a_dot_bt(lstm_states[time_step].delta_i, lstm_parameters.weights_xi);
    X_delta.add_a_dot_bt(lstm_states[time_step].delta_f, lstm_parameters.weights_xf);
    X_delta.add_a_dot_bt(lstm_states[time_step].delta_o, lstm_parameters.weights_xo);
    X_delta.add_a_dot_bt(lstm_states[time_step].delta_g, lstm_parameters.weights_xg);
}

template<unsigned long input_size, unsigned long mem_cell_size>
inline void update_weights_lstm(const Matrix<1,input_size> &X, std::vector<LstmState<input_size,mem_cell_size>> &lstm_states, LstmParam<input_size,mem_cell_size> &lstm_parameters, size_t time_step, double learning_rate)
{
    assert(time_step<lstm_states.size());
    lstm_states[time_step].delta_g.mul(learning_rate);
    lstm_states[time_step].delta_i.mul(learning_rate);
    lstm_states[time_step].delta_f.mul(learning_rate);
    lstm_states[time_step].delta_o.mul(learning_rate);

    lstm_parameters.weights_xg.add_at_dot_b(X, lstm_states[time_step].delta_g);
    lstm_parameters.weights_xi.add_at_dot_b(X, lstm_states[time_step].delta_i);
    lstm_parameters.weights_xf.add_at_dot_b(X, lstm_states[time_step].delta_f);
    lstm_parameters.weights_xo.add_at_dot_b(X, lstm_states[time_step].delta_o);
    if(time_step!=0)
    {
        lstm_parameters.weights_hg.add_at_dot_b(lstm_states[time_step-1].state_h, lstm_states[time_step].delta_g);
        lstm_parameters.weights_hi.add_at_dot_b(lstm_states[time_step-1].state_h, lstm_states[time_step].delta_i);
        lstm_parameters.weights_hf.add_at_dot_b(lstm_states[time_step-1].state_h, lstm_states[time_step].delta_f);
        lstm_parameters.weights_ho.add_at_dot_b(lstm_states[time_step-1].state_h, lstm_states[time_step].delta_o);
    }
    lstm_parameters.bias_g.add(lstm_states[time_step].delta_g);
    lstm_parameters.bias_i.add(lstm_states[time_step].delta_i);
    lstm_parameters.bias_f.add(lstm_states[time_step].delta_f);
    lstm_parameters.bias_o.add(lstm_states[time_step].delta_o);
}

/*
     _____           _    ______                       _                   
    |_   _|         | |   | ___ \                     | |                  
      | | __ _ _ __ | |__ | |_/ /__ _ __ ___ ___ _ __ | |_ _ __ ___  _ __  
      | |/ _` | '_ \| '_ \|  __/ _ \ '__/ __/ _ \ '_ \| __| '__/ _ \| '_ \ 
      | | (_| | | | | | | | | |  __/ | | (_|  __/ |_) | |_| | | (_) | | | |
      \_/\__,_|_| |_|_| |_\_|  \___|_|  \___\___| .__/ \__|_|  \___/|_| |_|
                                                | |                        
                                                |_|                        
*/

template<unsigned long input_size, unsigned long output_size>
inline void calc_layer_tanh(const Matrix<1,input_size> &X, std::vector<LayerState<input_size,output_size>> &layer_states, const LayerParam<input_size,output_size> &layer_parameters, size_t time_step)
{
    assert(time_step<layer_states.size());
    layer_states[time_step].output.equals_a_dot_b(X, layer_parameters.weights);
    layer_states[time_step].output.add(layer_parameters.bias);
    layer_states[time_step].output.apply_tanh();
}

template<unsigned long input_size, unsigned long output_size>
inline void set_first_delta_layer_tanh(const Matrix<1,output_size> &Y, std::vector<LayerState<input_size,output_size>> &layer_states, size_t time_step)
{
    assert(time_step<layer_states.size());
    //Get outputs delta
    layer_states[time_step].delta_output.equals_a_sub_b(Y,layer_states[time_step].output);
}

template<unsigned long input_size, unsigned long output_size>
inline void propagate_delta_layer_tanh(Matrix<1,input_size> &X_delta, std::vector<LayerState<input_size,output_size>> &layer_states, const LayerParam<input_size,output_size> &layer_parameters, size_t time_step)
{
    assert(time_step<layer_states.size());
    //Propagate delta to next building block
    layer_states[time_step].delta_output.mult_after_func02(layer_states[time_step].output);
    X_delta.equals_a_dot_bt(layer_states[time_step].delta_output, layer_parameters.weights);
}

template<unsigned long input_size, unsigned long output_size>
inline void set_first_delta_and_propagate_layer_tanh(const Matrix<1,output_size> &Y, Matrix<1,input_size> &X_delta, std::vector<LayerState<input_size,output_size>> &layer_states, const LayerParam<input_size,output_size> &layer_parameters, size_t time_step)
{
    assert(time_step<layer_states.size());
    //Get outputs delta
    layer_states[time_step].delta_output.equals_a_sub_b(Y,layer_states[time_step].output);
    layer_states[time_step].delta_output.mult_after_func02(layer_states[time_step].output);
    //Propagate to next building block
    X_delta.equals_a_dot_bt(layer_states[time_step].delta_output, layer_parameters.weights);
}

template<unsigned long input_size, unsigned long output_size>
inline void update_weights_layer_tanh(const Matrix<1,input_size> &X, std::vector<LayerState<input_size,output_size>> &layer_states, LayerParam<input_size,output_size> &layer_parameters, size_t time_step, double learning_rate)
{
    assert(time_step<layer_states.size());
    layer_states[time_step].delta_output.mul(learning_rate);
    layer_parameters.weights.add_at_dot_b(X, layer_states[time_step].delta_output);
    layer_parameters.bias.add(layer_states[time_step].delta_output);
}