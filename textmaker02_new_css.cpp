#include "perceptron_timeseries_class.hpp"
#include "LSTM_class.hpp"
#include "softmax_timeseries_class.hpp"
#include <unordered_map>
#include <chrono>
#include <memory>
using namespace std;




int main()
{
    const char *savecounter_filename="data/savecounter.svc";
    const char *savestate_filename="data/savestate%08lu.sst";
    const char *wb_filename="data/wb.wab";
    const char *output_filename="data/output%08lu.txt";
    const char *errors_filename="data/errors.txt";
    const char *asoiaf_filename="../allcss/allcss.css";
    static constexpr size_t output_length=2000;
    static constexpr size_t output_quantity=5;

    static constexpr size_t allowed_char_amount=72;
    static constexpr size_t min_training_chars=200;
    static constexpr size_t max_training_chars=1000;
    static constexpr auto time_between_saves=chrono::hours{6};
    static constexpr auto time_between_error_saves=chrono::minutes{20};

    static constexpr unsigned long input_size=allowed_char_amount;
    static constexpr unsigned long reduced_input_size=allowed_char_amount/4;
    static constexpr unsigned long first_mem_cell_size=150;
    static constexpr unsigned long second_mem_cell_size=150;
    static constexpr unsigned long third_mem_cell_size=300;
    static constexpr unsigned long output_mem_size=allowed_char_amount;
    static constexpr double decay=0.9;
    static constexpr size_t batch_size=10;

    auto get_learning_rate=[](size_t iteration){
        return (0.05/batch_size)*pow(0.9772372209558107, ((iteration*batch_size)/1000.));
        // return (0.05/batch_size)*1./(1.+0.00004*(iteration*batch_size))
    };
    double learning_rate=get_learning_rate(0);


    using Block01=RMSPropTahnPerceptronBlock<input_size,reduced_input_size>;
    using Block02=RMSPropLSTMBlock<reduced_input_size, first_mem_cell_size>;
    using Block03=RMSPropLSTMBlock<first_mem_cell_size, second_mem_cell_size>;
    using Block04=RMSPropLSTMBlock<second_mem_cell_size, third_mem_cell_size>;
    using Block05=RMSPropSoftmaxBlock<third_mem_cell_size, output_mem_size>;

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
    const string index_to_char="! #\"%$'&)(+*-,/.1032547698;:=<?>@[]\\_^a`cbedgfihkjmlonqpsrutwvyx{z}|~\n\r\t";
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
            learning_rate=get_learning_rate(iteration);
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
            print("First run... starting pre-training.");
            using BlockPre=RMSPropSoftmaxBlock<reduced_input_size,input_size>;
            unique_ptr<BlockPre> softmaxblock2(new BlockPre);
            softmaxblock2->set_time_steps(1);
            perceptronblock->set_time_steps(1);
            std::uniform_int_distribution<size_t> dst(0,input_size-1);

            double error=1.0;
            for(size_t iteration=0;error>0.00000001;iteration++)
            {
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
                perceptronblock->update_weights_ms(learning_rate, decay);
                softmaxblock2->update_weights_ms(learning_rate, decay);

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

    auto last_time=chrono::steady_clock::now();
    auto last_error_time=chrono::steady_clock::now();
    //All of the training happens within this loop
    for(;;iteration++)
    {
        learning_rate=get_learning_rate(iteration);
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
            // new_error/=batch_size;
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
        perceptronblock->update_weights_ms(learning_rate, decay);
        lstmblock1->update_weights_ms(learning_rate, decay);
        lstmblock2->update_weights_ms(learning_rate, decay);
        lstmblock3->update_weights_ms(learning_rate, decay);
        softmaxblock->update_weights_ms(learning_rate, decay);

        auto current_time=chrono::steady_clock::now();
        if(current_time-last_error_time>time_between_error_saves)
        {
            last_error_time=current_time;
            print("Saving error...");
            ofstream out(errors_filename, std::fstream::app);
            // auto actual_error=iteration*batch_size>1000?error:error*(1000./(iteration*batch_size>0?iteration*batch_size:1.));
            out << iteration << "\t" << error << "\n";
        }

        if(current_time-last_time>time_between_saves)
        {
            last_time=current_time;
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
                out << "Optimizer: " << "rmsprop" << endl;
                out << "Used memory cells: " << first_mem_cell_size << ", " << second_mem_cell_size << ", " << third_mem_cell_size << endl;
                out << "Batch size: " << batch_size << endl;
                out << "Learning rate: " << learning_rate << endl;
                out << "Decay: " << decay << endl;
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