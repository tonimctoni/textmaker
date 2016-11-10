#include "perceptron_timeseries_class.hpp"
#include "RELULSTM_class.hpp"
#include "softmax_timeseries_class.hpp"
#include <unordered_map>
#include <ctime>
#include <memory>
using namespace std;




int main()
{
    const char *savecounter_filename="data/savecounter.svc";
    const char *savestate_filename="data/savestate%08lu.sst";
    const char *wb_filename="data/wb.wab";
    const char *output_filename="data/output%08lu.txt";
    const char *asoiaf_filename="../asoiaf/asoiaf.txt";
    static constexpr size_t output_length=2000;
    static constexpr size_t output_quantity=5;

    static constexpr size_t allowed_char_amount=46;
    static constexpr size_t min_training_chars=100;
    static constexpr size_t max_training_chars=500;
    static constexpr time_t secons_between_saves=6*60*60;

    static constexpr unsigned long input_size=allowed_char_amount;
    static constexpr unsigned long reduced_input_size=allowed_char_amount/4;
    static constexpr unsigned long first_mem_cell_size=512;
    static constexpr unsigned long second_mem_cell_size=256;
    static constexpr unsigned long third_mem_cell_size=128;
    static constexpr unsigned long output_mem_size=allowed_char_amount;
    double learning_rate=0.01;
    double momentum=0.5;
    static constexpr size_t batch_size=1;


    using Block01=NAGTahnPerceptronBlock<input_size,reduced_input_size>;
    using Block02=NAGRELULSTMBlock<reduced_input_size, first_mem_cell_size>;
    using Block03=NAGRELULSTMBlock<first_mem_cell_size, second_mem_cell_size>;
    using Block04=NAGRELULSTMBlock<second_mem_cell_size, third_mem_cell_size>;
    using Block05=NAGSoftmaxBlock<third_mem_cell_size, output_mem_size>;

    unique_ptr<Block01> perceptronblock(new Block01);
    unique_ptr<Block02> lstmblock1(new Block02);
    unique_ptr<Block03> lstmblock2(new Block03);
    unique_ptr<Block04> lstmblock3(new Block04);
    unique_ptr<Block05> softmaxblock(new Block05);

    perceptronblock->reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);
    lstmblock1->reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);
    lstmblock2->reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);
    lstmblock3->reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);
    softmaxblock->reserve_time_steps(max_training_chars>output_length?max_training_chars:output_length);


    //Setup the char_to_index and index_to_char mappings
    const string index_to_char="! ')(-,.103254769;:?acbedgfihkjmlonqpsrutwvyxz";
    unordered_map<char, size_t> char_to_index;for(size_t i=0;i<index_to_char.size();i++) char_to_index[index_to_char[i]]=i;
    assert(index_to_char.size()==allowed_char_amount and char_to_index.size()==allowed_char_amount);
    //Read the text file
    string asoiaf_content;
    read_file_to_string(asoiaf_filename, asoiaf_content);

    //This matrix will be used as input
    OneHot<input_size> X;
    OneHot<output_mem_size> Y;

    //Initialize random number generator and distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dst_start(1,asoiaf_content.size()-max_training_chars-1);
    std::uniform_int_distribution<size_t> dst_lengt(min_training_chars, max_training_chars);
    std::uniform_int_distribution<size_t> dst_rdchar(0, allowed_char_amount-1);

    //Retrieve the number of the last saved state and last iteration (if exists) and the saved state with this number
    double error=0;
    size_t save_counter=0;
    size_t iteration=0;
    {
        ifstream in(savecounter_filename, std::ios::binary);
        if(in.good())
        {
            in >> save_counter >> iteration >> error;
            assert(save_counter!=0);
            {
                char cbuffer[256];
                sprintf(cbuffer, savestate_filename, save_counter);
                ifstream in(cbuffer);
                assert(in.good());
                perceptronblock->from_bin_file(in);
                lstmblock1->from_bin_file(in);
                lstmblock2->from_bin_file(in);
                lstmblock3->from_bin_file(in);
                softmaxblock->from_bin_file(in);
            }
        }
        else
        {
            static constexpr double learning_rate=0.1;
            static constexpr double momentum=0.9;
            static constexpr size_t batch_size=1;
            print("First run... starting pre-training.");
            using BlockPre=NAGSoftmaxBlock<reduced_input_size,input_size>;
            unique_ptr<BlockPre> softmaxblock2(new BlockPre);
            softmaxblock2->set_time_steps(1);
            perceptronblock->set_time_steps(1);
            std::uniform_int_distribution<size_t> dst(0,input_size-1);

            double error=1.0;
            for(size_t iteration=0;error>0.00000001;iteration++)
            {
                perceptronblock->apply_momentum(momentum);
                softmaxblock2->apply_momentum(momentum);
                for(size_t batch=0;batch<batch_size;batch++)
                {
                    X.set(dst(gen));

                    perceptronblock->calc(X.get(), 0);
                    softmaxblock2->calc(perceptronblock->get_output(0), 0);

                    softmaxblock2->set_first_delta_and_propagate_with_cross_enthropy(X.get(), perceptronblock->get_delta_output(0), 0);
                    perceptronblock->propagate_delta(0);

                    perceptronblock->accumulate_gradients(X.get(),0);
                    softmaxblock2->accumulate_gradients(perceptronblock->get_output(0),0);
                }
                perceptronblock->update_weights_momentum(learning_rate);
                softmaxblock2->update_weights_momentum(learning_rate);

                if(iteration%10000==0)
                {
                    error=0.0;
                    for(size_t new_input_index=0;new_input_index<input_size;new_input_index++)
                    {
                        X.set(dst(gen));

                        perceptronblock->calc(X.get(), 0);
                        softmaxblock2->calc(perceptronblock->get_output(0), 0);

                        for(size_t i=0;i<input_size;i++)
                        {
                            double aux=softmaxblock2->get_output(0)[0][i]-X.get()[0][i];
                            aux*=aux;
                            error+=aux;
                        }
                    }
                    // error/=input_size;
                    print(error);
                }
            }
        }
    }

    time_t last_time=time(nullptr);
    time_t first_time=last_time;
    //All of the training happens within this loop
    for(;;iteration++)
    {
        if(iteration%1000==0)
        {
            learning_rate=0.01*pow(0.9549925860214360, (iteration/1000));// gets divided by 10 every 50k steps
            if(iteration<=50000) momentum=0.5+0.008*(iteration/1000);
            else if(iteration<=100000)momentum=0.9+0.0018*((iteration-50000)/1000);
            // else if(iteration<=150000)momentum=0.99+0.00018*((iteration-100000)/1000);
            // else if(iteration<=200000)momentum=0.999+0.000018*((iteration-150000)/1000);
            else momentum=.99;
        }
        perceptronblock->apply_momentum(momentum);
        lstmblock1->apply_momentum(momentum);
        lstmblock2->apply_momentum(momentum);
        lstmblock3->apply_momentum(momentum);
        softmaxblock->apply_momentum(momentum);
        for(size_t batch=0;batch<batch_size;batch++)
        {
            //Pick what characters to feed to the lsmts
            size_t asoiaf_start=dst_start(gen);
            size_t asoiaf_length=dst_lengt(gen);
            perceptronblock->set_time_steps(asoiaf_length);
            lstmblock1->set_time_steps(asoiaf_length);
            lstmblock2->set_time_steps(asoiaf_length);
            lstmblock3->set_time_steps(asoiaf_length);
            softmaxblock->set_time_steps(asoiaf_length);

            //Let the network calculate an output given the input
            for(size_t i=0;i<asoiaf_length;i++)
            {
                //Set up input
                X.set(char_to_index[asoiaf_content[asoiaf_start+i-1]]);

                perceptronblock->calc(X.get(), i);
                lstmblock1->calc(perceptronblock->get_output(i), i);
                lstmblock2->calc(lstmblock1->get_output(i), i);
                lstmblock3->calc(lstmblock2->get_output(i), i);
                softmaxblock->calc(lstmblock3->get_output(i), i);
            }

            double new_error=0;
            //Calculate deltas
            for(size_t i=asoiaf_length-1;;)
            {
                //Set up output
                Y.set(char_to_index[asoiaf_content[asoiaf_start+i]]);

                softmaxblock->set_first_delta_and_propagate_with_cross_enthropy(Y.get(), lstmblock3->get_delta_output(i), i);
                new_error+=softmaxblock->get_delta_output(i).sum_of_squares();
                lstmblock3->propagate_delta(lstmblock2->get_delta_output(i), i, asoiaf_length);
                lstmblock2->propagate_delta(lstmblock1->get_delta_output(i), i, asoiaf_length);
                lstmblock1->propagate_delta(perceptronblock->get_delta_output(i), i, asoiaf_length);
                perceptronblock->propagate_delta(i);
                if(i--==0)break;
            }
            new_error/=batch_size;
            new_error/=asoiaf_length;
            error*=0.999;
            error+=new_error*0.001;

            //Update gradients
            for(size_t i=0;i<asoiaf_length;i++)
            {
                //Set up input
                X.set(char_to_index[asoiaf_content[asoiaf_start+i-1]]);

                perceptronblock->accumulate_gradients(X.get(), i);
                lstmblock1->accumulate_gradients(perceptronblock->get_output(i), i);
                lstmblock2->accumulate_gradients(lstmblock1->get_output(i), i);
                lstmblock3->accumulate_gradients(lstmblock2->get_output(i), i);
                softmaxblock->accumulate_gradients(lstmblock3->get_output(i), i);
            }
        }
        perceptronblock->update_weights_momentum(learning_rate);
        lstmblock1->update_weights_momentum(learning_rate);
        lstmblock2->update_weights_momentum(learning_rate);
        lstmblock3->update_weights_momentum(learning_rate);
        softmaxblock->update_weights_momentum(learning_rate);


        if(time(nullptr)-last_time>secons_between_saves)
        {
            last_time=time(nullptr);
            print("Saving current state...");
            save_counter++;
            char cbuffer[256];
            //Save weights to file
            sprintf(cbuffer, savestate_filename, save_counter);
            {
                ofstream out(cbuffer,std::ios_base::trunc|std::ios::binary);
                assert(out.good());
                perceptronblock->to_bin_file(out);
                lstmblock1->to_bin_file(out);
                lstmblock2->to_bin_file(out);
                lstmblock3->to_bin_file(out);
                softmaxblock->to_bin_file(out);
            }
            {
                ofstream out(wb_filename,std::ios_base::trunc|std::ios::binary);
                assert(out.good());
                perceptronblock->only_wb_to_bin_file(out);
                lstmblock1->only_wb_to_bin_file(out);
                lstmblock2->only_wb_to_bin_file(out);
                lstmblock3->only_wb_to_bin_file(out);
                softmaxblock->only_wb_to_bin_file(out);
            }
            //Save some data and some output examples to file
            sprintf(cbuffer, output_filename, save_counter);
            {
                ofstream out(cbuffer,std::ios_base::trunc);
                assert(out.good());
                out << "Current iteration: " << iteration << endl;
                out << "Seconds elapsed since program started: " << time(nullptr)-first_time << endl;
                out << "Optimizer: " << "nag" << endl;
                out << "Used memory cells: " << first_mem_cell_size << ", " << second_mem_cell_size << ", " << third_mem_cell_size << endl;
                out << "Batch size: " << batch_size << endl;
                out << "Learning rate: " << learning_rate << endl;
                out << "Momentum: " << momentum << endl;
                out << "Error: " << error << endl;
                out << endl;
                out << endl;
                out << endl;
                out << endl;
                for(size_t ex=0;ex<output_quantity;ex++)
                {
                    out << "################################ EXAMPLE " << ex+1 << " ################################" << endl;
                    size_t next_input_index=dst_rdchar(gen);
                    perceptronblock->set_time_steps(output_length);
                    lstmblock1->set_time_steps(output_length);
                    lstmblock2->set_time_steps(output_length);
                    lstmblock3->set_time_steps(output_length);
                    softmaxblock->set_time_steps(output_length);
                    for(size_t i=0;i<output_length;i++)
                    {
                        X.set(next_input_index);

                        perceptronblock->calc(X.get(), i);
                        lstmblock1->calc(perceptronblock->get_output(i), i);
                        lstmblock2->calc(lstmblock1->get_output(i), i);
                        lstmblock3->calc(lstmblock2->get_output(i), i);
                        softmaxblock->calc(lstmblock3->get_output(i), i);
                        next_input_index=get_weighted_random_index(softmaxblock->get_output(i)[0]);
                        out << index_to_char[next_input_index];
                    }
                    out << endl;
                    out << endl;
                    out << endl;
                    out << endl;
                }
            }
            {
                ofstream out(savecounter_filename,std::ios_base::trunc);
                assert(out.good());
                out << save_counter << "\t" << iteration << "\t" << error << endl;
            }
            print("State with number", save_counter, "saved");
        }
        print("Iteration:", iteration, "Error:", error);
    }
    return 0;
}