#include "matrix.hpp"
#include <vector>

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
    //Deltas
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

template<unsigned long input_size, unsigned long mem_cell_size>
class BaseLSTMBlock
{
protected:
    std::vector<LstmState<input_size, mem_cell_size>> lstm_states;
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

    //LSTM gradient accumulator for weights
    Matrix<input_size, mem_cell_size> weights_xg_gradient_acc;
    Matrix<input_size, mem_cell_size> weights_xi_gradient_acc;
    Matrix<input_size, mem_cell_size> weights_xf_gradient_acc;
    Matrix<input_size, mem_cell_size> weights_xo_gradient_acc;
    Matrix<mem_cell_size, mem_cell_size> weights_hg_gradient_acc;
    Matrix<mem_cell_size, mem_cell_size> weights_hi_gradient_acc;
    Matrix<mem_cell_size, mem_cell_size> weights_hf_gradient_acc;
    Matrix<mem_cell_size, mem_cell_size> weights_ho_gradient_acc;
    //LSTM gradient accumulator for biases
    Matrix<1, mem_cell_size> bias_g_gradient_acc;
    Matrix<1, mem_cell_size> bias_i_gradient_acc;
    Matrix<1, mem_cell_size> bias_f_gradient_acc;
    Matrix<1, mem_cell_size> bias_o_gradient_acc;
public:
    //Constructor: randomizes the weights and biases
    BaseLSTMBlock(size_t time_steps=0) noexcept:lstm_states(time_steps)
    , weights_xg_gradient_acc(0.0), weights_xi_gradient_acc(0.0), weights_xf_gradient_acc(0.0), weights_xo_gradient_acc(0.0)
    , weights_hg_gradient_acc(0.0), weights_hi_gradient_acc(0.0), weights_hf_gradient_acc(0.0), weights_ho_gradient_acc(0.0)
    , bias_g_gradient_acc(0.0), bias_i_gradient_acc(0.0), bias_f_gradient_acc(0.0), bias_o_gradient_acc(0.0)
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
    }

    inline void only_wb_to_bin_file(std::ofstream &out) noexcept
    {
        weights_xg.to_bin_file(out);
        weights_xi.to_bin_file(out);
        weights_xf.to_bin_file(out);
        weights_xo.to_bin_file(out);
        weights_hg.to_bin_file(out);
        weights_hi.to_bin_file(out);
        weights_hf.to_bin_file(out);
        weights_ho.to_bin_file(out);
        bias_g.to_bin_file(out);
        bias_i.to_bin_file(out);
        bias_f.to_bin_file(out);
        bias_o.to_bin_file(out);
    }

    inline void only_wb_from_bin_file(std::ifstream &in) noexcept
    {
        weights_xg.from_bin_file(in);
        weights_xi.from_bin_file(in);
        weights_xf.from_bin_file(in);
        weights_xo.from_bin_file(in);
        weights_hg.from_bin_file(in);
        weights_hi.from_bin_file(in);
        weights_hf.from_bin_file(in);
        weights_ho.from_bin_file(in);
        bias_g.from_bin_file(in);
        bias_i.from_bin_file(in);
        bias_f.from_bin_file(in);
        bias_o.from_bin_file(in);
    }

    inline void set_time_steps(size_t time_steps) noexcept
    {
        lstm_states.resize(time_steps);
    }

    inline void reserve_time_steps(size_t time_steps) noexcept
    {
        lstm_states.reserve(time_steps);
    }

    inline void calc(const Matrix<1,input_size> &X, size_t time_step) noexcept
    {
        assert(time_step<lstm_states.size());
        //Calculate states g, i (input gate), f (forget gate), and o (output gate). steps are split for readability
        //f does not need to be calculated in the first round (nothing to forget there). maybe optimize that later.

        //Multiply input with corresponding weights for each state
        lstm_states[time_step].state_g.equals_a_dot_b(X,weights_xg);
        lstm_states[time_step].state_i.equals_a_dot_b(X,weights_xi);
        lstm_states[time_step].state_f.equals_a_dot_b(X,weights_xf);
        lstm_states[time_step].state_o.equals_a_dot_b(X,weights_xo);
        if(time_step!=0)
        {
            //Multiply last h-state with corresponding weights for each state
            lstm_states[time_step].state_g.add_a_dot_b(lstm_states[time_step-1].state_h, weights_hg);
            lstm_states[time_step].state_i.add_a_dot_b(lstm_states[time_step-1].state_h, weights_hi);
            lstm_states[time_step].state_f.add_a_dot_b(lstm_states[time_step-1].state_h, weights_hf);
            lstm_states[time_step].state_o.add_a_dot_b(lstm_states[time_step-1].state_h, weights_ho);
        }
        //Add biases to each state
        lstm_states[time_step].state_g.add(bias_g);
        lstm_states[time_step].state_i.add(bias_i);
        lstm_states[time_step].state_f.add(bias_f);
        lstm_states[time_step].state_o.add(bias_o);
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

    inline void set_first_delta(const Matrix<1,mem_cell_size> &Y, size_t time_step) noexcept
    {
        assert(time_step<lstm_states.size());
        //Get outputs delta
        lstm_states[time_step].delta_h.equals_a_sub_b(Y,lstm_states[time_step].state_h);
    }

    inline void propagate_delta(size_t time_step, size_t total_time_steps) noexcept
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

            lstm_states[time_step].delta_lh.equals_a_dot_bt(lstm_states[time_step].delta_i, weights_hi);
            lstm_states[time_step].delta_lh.add_a_dot_bt(lstm_states[time_step].delta_f, weights_hf);
            lstm_states[time_step].delta_lh.add_a_dot_bt(lstm_states[time_step].delta_o, weights_ho);
            lstm_states[time_step].delta_lh.add_a_dot_bt(lstm_states[time_step].delta_g, weights_hg);
        }
    }

    inline void propagate_delta(Matrix<1,input_size> &X_delta, size_t time_step, size_t total_time_steps) noexcept
    {
        propagate_delta(time_step, total_time_steps);

        X_delta.equals_a_dot_bt(lstm_states[time_step].delta_i, weights_xi);
        X_delta.add_a_dot_bt(lstm_states[time_step].delta_f, weights_xf);
        X_delta.add_a_dot_bt(lstm_states[time_step].delta_o, weights_xo);
        X_delta.add_a_dot_bt(lstm_states[time_step].delta_g, weights_xg);
    }

    inline void accumulate_gradients(const Matrix<1,input_size> &X, size_t time_step) noexcept
    {
        assert(time_step<lstm_states.size());

        weights_xg_gradient_acc.add_at_dot_b(X, lstm_states[time_step].delta_g);
        weights_xi_gradient_acc.add_at_dot_b(X, lstm_states[time_step].delta_i);
        weights_xf_gradient_acc.add_at_dot_b(X, lstm_states[time_step].delta_f);
        weights_xo_gradient_acc.add_at_dot_b(X, lstm_states[time_step].delta_o);
        if(time_step!=0)
        {
            weights_hg_gradient_acc.add_at_dot_b(lstm_states[time_step-1].state_h, lstm_states[time_step].delta_g);
            weights_hi_gradient_acc.add_at_dot_b(lstm_states[time_step-1].state_h, lstm_states[time_step].delta_i);
            weights_hf_gradient_acc.add_at_dot_b(lstm_states[time_step-1].state_h, lstm_states[time_step].delta_f);
            weights_ho_gradient_acc.add_at_dot_b(lstm_states[time_step-1].state_h, lstm_states[time_step].delta_o);
        }
        bias_g_gradient_acc.add(lstm_states[time_step].delta_g);
        bias_i_gradient_acc.add(lstm_states[time_step].delta_i);
        bias_f_gradient_acc.add(lstm_states[time_step].delta_f);
        bias_o_gradient_acc.add(lstm_states[time_step].delta_o);
    }

    inline const Matrix<1,mem_cell_size>& get_output(size_t time_step) const noexcept
    {
        return lstm_states[time_step].state_h;
    }

    inline Matrix<1,mem_cell_size>& get_delta_output(size_t time_step) noexcept
    {
        return lstm_states[time_step].delta_h;
    }
};
