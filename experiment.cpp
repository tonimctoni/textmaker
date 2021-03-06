#include "perceptron_timeseries_class.hpp"
#include "RELU_timeseries_class.hpp"
#include "softmax_timeseries_class.hpp"
#include "LSTM_class.hpp"
#include <memory>
#include <chrono>
// #include <fstream>
using namespace std;
//try out adam with learning_rate schedule
//read adam paper
//add timestamp to error message

// 0.7943282347242815 --> *.1 every 10
// 0.8912509381337456 --> *.1 every 20
// 0.9120108393559098 --> *.1 every 25
// 0.9440608762859234 --> *.1 every 40
// 0.9549925860214360 --> *.1 every 50
// 0.9772372209558107 --> *.1 every 100
int main()
{
    // std::ifstream in(filename, std::ios::in);
    // {
    //     std::ofstream out("example.txt",std::ios_base::trunc);
    //     assert(out.good());
    //     out << 2 << "\t" << 3 << endl;
    //     out << .25 << endl;
    //     out << .325 << endl;
    //     out << .425 << endl;
    //     out << .525 << endl;
    //     out << .625 << endl;
    //     out << .725 << endl;
    // }

    // {
    //     std::ifstream in("example.txt");
    //     assert(in.good());
    //     int a, b;
    //     double c;
    //     in >> a >> b;
    //     print(a,b);
    //     for(size_t i=0;i<a*b;i++)
    //     {
    //         in >> c;
    //         print(c);
    //     }
    // }

    // Matrix<2,4> m(1.0);
    // {
    //     std::ofstream out("example.txt",std::ios_base::trunc);
    //     m.to_file(out);
    // }
    // {
    //     std::ifstream in("example.txt");
    //     m.from_file(in);
    // }
    // print(m);

    // {
    //     LstmParam<2, 5> lstm_params;
    //     LayerParam<5,1> layer_params;
    //     lstm_params.weights_xg.set(11.12);
    //     lstm_params.weights_xi.set(11.12);
    //     lstm_params.weights_xf.set(11.12);
    //     lstm_params.weights_xo.set(11.12);
    //     lstm_params.weights_hg.set(11.12);
    //     lstm_params.weights_hi.set(15.16);
    //     lstm_params.weights_hf.set(11.12);
    //     lstm_params.weights_ho.set(11.12);
    //     lstm_params.bias_g.set(13.14);
    //     lstm_params.bias_i.set(11.12);
    //     lstm_params.bias_f.set(11.12);
    //     lstm_params.bias_o.set(11.12);
    //     layer_params.weights.set(1.0);
    //     layer_params.bias.set(5.0);
    //     std::ofstream out("lstm_params_example_01.txt",std::ios_base::trunc);
    //     assert(out.good());
    //     lstm_params.to_file(out);
    //     layer_params.to_file(out);
    // }

    // {
    //     LstmParam<2, 5> lstm_params;
    //     LayerParam<5,1> layer_params;
    //     std::ifstream in("lstm_params_example_01.txt");
    //     assert(in.good());
    //     lstm_params.from_file(in);
    //     layer_params.from_file(in);
    //     print(lstm_params.weights_xg);
    //     print(lstm_params.weights_hi);
    //     print(lstm_params.bias_g);
    //     print(lstm_params.bias_f);
    //     print(layer_params.weights);
    //     print(layer_params.bias);
    // }


    // const char* my_filename="asd.txt";
    // std::ifstream in(my_filename);
    // const char *savecounter_filename="data/savecounter.svc";

    // TahnPerceptronBlock<2,3> block1(1);
    // TahnPerceptronBlock<3,1> block2(1);


    // unique_ptr<TahnPerceptronBlock<2,3>> block1(new TahnPerceptronBlock<2,3>(1));
    // unique_ptr<TahnPerceptronBlock<3,1>> block2(new TahnPerceptronBlock<3,1>(2));

    // vector<Matrix<1,2>> X{{{0,0},},{{1,0},},{{0,1},},{{1,1},},};
    // vector<Matrix<1,1>> Y{{{0, },},{{1, },},{{1, },},{{0, },},};
    // print("################################");
    // for(size_t i=0;i<10000000;i++)//10000000 -> 0m13.372s
    // {
    //     for(size_t j=0;j<4;j++)
    //     {
            // block1->calc(X[j], 0);
            // block2->calc(block1->get_output(0), 0);

            // block2->set_first_delta(Y[j], 0);
            // block2->propagate_delta(block1->get_delta_output(0), 0);
            // block1->propagate_delta(0);

            // block1->update_weights(X[j],0,0.2);
            // block2->update_weights(block1->get_output(0),0,0.2);
    //     }
    // }

    // for(size_t j=0;j<4;j++)
    // {
    //     block1->calc(X[j], 0);
    //     block2->calc(block1->get_output(0), 0);

    //     print(block2->get_output(0));
    // }

    // static constexpr size_t input_size=98;
    // static constexpr size_t hidden_size=25;
    // // static constexpr size_t hidden_size_a=80;
    // // static constexpr size_t hidden_size_b=15;

    // unique_ptr<TahnPerceptronBlock<input_size,hidden_size>> block1(new TahnPerceptronBlock<input_size,hidden_size>(1));
    // unique_ptr<SoftmaxBlock<hidden_size,input_size>> block2(new SoftmaxBlock<hidden_size,input_size>(1));

    // // unique_ptr<TahnPerceptronBlock<input_size,hidden_size_a>> block1a(new TahnPerceptronBlock<input_size,hidden_size_a>(1));
    // // unique_ptr<TahnPerceptronBlock<hidden_size_a,hidden_size_b>> block1b(new TahnPerceptronBlock<hidden_size_a,hidden_size_b>(1));

    // // unique_ptr<TahnPerceptronBlock<hidden_size_b,hidden_size_a>> block2b(new TahnPerceptronBlock<hidden_size_b,hidden_size_a>(1));
    // // unique_ptr<TahnPerceptronBlock<hidden_size_a,input_size>> block2a(new TahnPerceptronBlock<hidden_size_a,input_size>(1));




    // Matrix<1,input_size> X(0.0);
    // size_t last_input_index=0;
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<size_t> dst(0,input_size-1);

    // // double last_average_error=0.0;
    // for(size_t iteration=0;iteration<5000000;iteration++)
    // {
    //     X[0][last_input_index]=0.0;
    //     last_input_index=dst(gen);
    //     X[0][last_input_index]=1.0;

    //     block1->calc(X, 0);
    //     block2->calc(block1->get_output(0), 0);
    //     // block1a->calc(X, 0);
    //     // block1b->calc(block1a->get_output(0), 0);
    //     // block2b->calc(block1b->get_output(0), 0);
    //     // block2a->calc(block2b->get_output(0), 0);

    //     // block2->set_first_delta(X, 0);
    //     // block2->propagate_delta(block1->get_delta_output(0), 0);
    //     block2->set_first_delta_and_propagate_with_cross_enthropy(X, block1->get_delta_output(0), 0);
    //     block1->propagate_delta(0);
    //     // block2a->set_first_delta(X, 0);
    //     // block2a->propagate_delta(block2b->get_delta_output(0), 0);
    //     // block2b->propagate_delta(block1b->get_delta_output(0), 0);
    //     // block1b->propagate_delta(block1a->get_delta_output(0), 0);
    //     // block1a->propagate_delta(0);

    //     block1->update_weights(X,0,0.2);
    //     block2->update_weights(block1->get_output(0),0,0.2);
    //     // block1a->update_weights(X,0,0.2);
    //     // block1b->update_weights(block1a->get_output(0),0,0.2);
    //     // block2b->update_weights(block1b->get_output(0),0,0.2);
    //     // block2a->update_weights(block2b->get_output(0),0,0.2);

    //     if(iteration%1000==0)
    //     {
    //         double error=0.0;
    //         for(size_t new_input_index=0;new_input_index<input_size;new_input_index++)
    //         {
    //             X[0][last_input_index]=0.0;
    //             last_input_index=new_input_index;
    //             X[0][last_input_index]=1.0;

    //             block1->calc(X, 0);
    //             block2->calc(block1->get_output(0), 0);

    //             for(size_t i=0;i<input_size;i++)
    //             {
    //                 double aux=block2->get_output(0)[0][i]-X[0][i];
    //                 aux*=aux;
    //                 error+=aux;
    //             }
    //         }
    //         // error/=input_size;
    //         print(error);
    //     }
    // }

    // for(size_t new_input_index=0;new_input_index<input_size;new_input_index++)
    // {
    //     X[0][last_input_index]=0.0;
    //     last_input_index=new_input_index;
    //     X[0][last_input_index]=1.0;

    //     block1->calc(X, 0);
    //     block2->calc(block1->get_output(0), 0);

    //     print(block1->get_output(0));
    //     // for(size_t i=0;i<input_size;i++)
    //     // {
    //     //     double aux=block2->get_output(0)[0][i]-X[0][i];
    //     //     aux*=aux;
    //     //     error+=aux;
    //     // }
    // }


    // unique_ptr<TahnPerceptronBlock<2,3>> block1(new TahnPerceptronBlock<2,3>(1));
    // unique_ptr<TahnPerceptronBlock<3,1>> block2(new TahnPerceptronBlock<3,1>(2));

    // vector<Matrix<1,2>> X{{{0,0},},{{1,0},},{{0,1},},{{1,1},},};
    // vector<Matrix<1,1>> Y{{{0, },},{{1, },},{{1, },},{{0, },},};
    // print("################################");
    // for(size_t i=0;i<100000;i++)//10000000 -> 0m13.372s
    // {
    //     block1->apply_momentum(0.9);
    //     block2->apply_momentum(0.9);
    //     for(size_t j=0;j<4;j++)
    //     {
    //         block1->calc(X[j], 0);
    //         block2->calc(block1->get_output(0), 0);

    //         block2->set_first_delta(Y[j], 0);
    //         block2->propagate_delta(block1->get_delta_output(0), 0);
    //         block1->propagate_delta(0);

    //         block1->accumulate_gradients(X[j],0);
    //         block2->accumulate_gradients(block1->get_output(0),0);
    //     }
    //     block1->update_weights_momentum_ms(.001, .9);
    //     block2->update_weights_momentum_ms(.001, .9);
    // }

    // for(size_t j=0;j<4;j++)
    // {
    //     block1->calc(X[j], 0);
    //     block2->calc(block1->get_output(0), 0);

    //     print(block2->get_output(0));
    // }

    // unique_ptr<TahnPerceptronBlock<2,3>> block1(new TahnPerceptronBlock<2,3>(1));
    // unique_ptr<TahnPerceptronBlock<3,1>> block2(new TahnPerceptronBlock<3,1>(2));

    // vector<Matrix<1,2>> X{{{0,0},},{{1,0},},{{0,1},},{{1,1},},};
    // vector<Matrix<1,1>> Y{{{0, },},{{1, },},{{1, },},{{0, },},};
    // print("################################");
    // for(size_t i=0;i<100000;i++)//10000000 -> 0m13.372s
    // {
    //     for(size_t j=0;j<4;j++)
    //     {
    //         block1->calc(X[j], 0);
    //         block2->calc(block1->get_output(0), 0);

    //         block2->set_first_delta(Y[j], 0);
    //         block2->propagate_delta(block1->get_delta_output(0), 0);
    //         block1->propagate_delta(0);

    //         block1->accumulate_gradients(X[j],0);
    //         block2->accumulate_gradients(block1->get_output(0),0);
    //     }
    //     block1->update_weights_adam(.001, .999, .9);
    //     block2->update_weights_adam(.001, .999, .9);
    // }

    // for(size_t j=0;j<4;j++)
    // {
    //     block1->calc(X[j], 0);
    //     block2->calc(block1->get_output(0), 0);

    //     print(block2->get_output(0));
    // }


    // unique_ptr<NAGTahnPerceptronBlock<2,3>> block1(new NAGTahnPerceptronBlock<2,3>(1));
    // unique_ptr<NAGTahnPerceptronBlock<3,1>> block2(new NAGTahnPerceptronBlock<3,1>(2));

    // vector<Matrix<1,2>> X{{{0,0},},{{1,0},},{{0,1},},{{1,1},},};
    // vector<Matrix<1,1>> Y{{{0, },},{{1, },},{{1, },},{{0, },},};
    // print("################################");
    // for(size_t i=0;i<10000000;i++)//10000000 -> 0m13.372s
    // {
    //     block1->apply_momentum(.9);
    //     block2->apply_momentum(.9);
    //     for(size_t j=0;j<4;j++)
    //     {
    //         block1->calc(X[j], 0);
    //         block2->calc(block1->get_output(0), 0);

    //         block2->set_first_delta(Y[j], 0);
    //         block2->propagate_delta(block1->get_delta_output(0), 0);
    //         block1->propagate_delta(0);

    //         block1->accumulate_gradients(X[j],0);
    //         block2->accumulate_gradients(block1->get_output(0),0);
    //     }
    //     block1->update_weights_momentum(.1);
    //     block2->update_weights_momentum(.1);
    // }

    // for(size_t j=0;j<4;j++)
    // {
    //     block1->calc(X[j], 0);
    //     block2->calc(block1->get_output(0), 0);

    //     print(block2->get_output(0));
    // }







    // static constexpr size_t input_size=6;
    // static constexpr size_t hidden_size=1;

    // unique_ptr<AdamTahnPerceptronBlock<input_size,hidden_size>> block1(new AdamTahnPerceptronBlock<input_size,hidden_size>(1));
    // unique_ptr<AdamSoftmaxBlock<hidden_size,input_size>> block2(new AdamSoftmaxBlock<hidden_size,input_size>(1));

    // Matrix<1,input_size> X(0.0);
    // size_t last_input_index=0;
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<size_t> dst(0,input_size-1);

    // // double last_average_error=0.0;
    // for(size_t iteration=0;iteration<5000000;iteration++)
    // {
    //     for(size_t batch=0;batch<1;batch++)
    //     {
    //         X[0][last_input_index]=0.0;
    //         last_input_index=dst(gen);
    //         X[0][last_input_index]=1.0;

    //         block1->calc(X, 0);
    //         block2->calc(block1->get_output(0), 0);

    //         block2->set_first_delta_and_propagate_with_cross_enthropy(X, block1->get_delta_output(0), 0);
    //         block1->propagate_delta(0);

    //         block1->accumulate_gradients(X,0);
    //         block2->accumulate_gradients(block1->get_output(0),0);
    //     }
    //     block1->update_weights_adam(.001, .999, .9);
    //     block2->update_weights_adam(.001, .999, .9);

    //     if(iteration%100000==0)
    //     {
    //         double error=0.0;
    //         for(size_t new_input_index=0;new_input_index<input_size;new_input_index++)
    //         {
    //             X[0][last_input_index]=0.0;
    //             last_input_index=new_input_index;
    //             X[0][last_input_index]=1.0;

    //             block1->calc(X, 0);
    //             block2->calc(block1->get_output(0), 0);

    //             for(size_t i=0;i<input_size;i++)
    //             {
    //                 double aux=block2->get_output(0)[0][i]-X[0][i];
    //                 aux*=aux;
    //                 error+=aux;
    //             }
    //         }
    //         // error/=input_size;
    //         print(error);
    //     }
    // }
    // print("############");
    // for(size_t i=0;i<input_size;i++)
    // {
    //     X[0][last_input_index]=0.0;
    //     last_input_index=i;
    //     X[0][last_input_index]=1.0;

    //     block1->calc(X, 0);
    //     // block2->calc(block1->get_output(0), 0);
    //     print(block1->get_output(0));
    // }


    // string asoiaf_content;
    // read_file_to_string("../asoiaf/asoiaf.txt", asoiaf_content);
    // vector<size_t> sentence_starts;
    // sentence_starts.reserve(150000);
    // sentence_starts.push_back(0);
    // for(size_t i=0;i<asoiaf_content.size()-1;i++)
    //     if(asoiaf_content[i]=='.' and asoiaf_content[i+1]!='.' and asoiaf_content[i-1]!='.')
    //         sentence_starts.push_back(i+1);
    // sentence_starts.push_back(asoiaf_content.size());

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<size_t> dst_start(0,sentence_starts.size()-2);

    // size_t start=dst_start(gen);
    // for(size_t i=sentence_starts[start];i<sentence_starts[start+1];i++)
    //     cout << asoiaf_content[i];
    // cout << endl;
    // print(asoiaf_content.size());
    // print(sentence_starts.size());

    // unique_ptr<SGDRELUBlock<4,5>> block1(new SGDRELUBlock<4,5>(1));
    // unique_ptr<SGDRELUBlock<5,2>> block2(new SGDRELUBlock<5,2>(2));

    // vector<Matrix<1,4>> X{{{0,0,0,0},},{{1,0,1,0},},{{0,1,0,1},},{{1,1,1,1},},};
    // vector<Matrix<1,2>> Y{{{0,0},},{{1,1},},{{1,1},},{{0,0},},};
    // print("################################");
    // for(size_t i=0;i<100000;i++)//10000000 -> 0m13.372s
    // {
    //     // print(i);
    //     // block1->apply_momentum(.9);
    //     // block2->apply_momentum(.9);
    //     for(size_t j=0;j<4;j++)
    //     {
    //         block1->calc(X[j], 0);
    //         block2->calc(block1->get_output(0), 0);

    //         block2->set_first_delta(Y[j], 0);
    //         block2->propagate_delta(block1->get_delta_output(0), 0);
    //         block1->propagate_delta(0);

    //         block1->accumulate_gradients(X[j],0);
    //         block2->accumulate_gradients(block1->get_output(0),0);
    //     }
    //     // block1->update_weights_momentum(.1);
    //     // block2->update_weights_momentum(.1);
    //     // block1->update_weights_ms(.1, .9);
    //     // block2->update_weights_ms(.1, .9);
    //     block1->update_weights(.8);
    //     block2->update_weights(.8);
    // }

    // for(size_t j=0;j<4;j++)
    // {
    //     block1->calc(X[j], 0);
    //     block2->calc(block1->get_output(0), 0);

    //     print(block2->get_output(0));
    // }

    // static constexpr size_t input_size=4;
    // static constexpr size_t hidden_size=1;

    // unique_ptr<AdamRELUBlock<input_size,hidden_size>> block1(new AdamRELUBlock<input_size,hidden_size>(1));
    // unique_ptr<AdamSoftmaxBlock<hidden_size,input_size>> block2(new AdamSoftmaxBlock<hidden_size,input_size>(1));

    // Matrix<1,input_size> X(0.0);
    // size_t last_input_index=0;
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<size_t> dst(0,input_size-1);

    // // double last_average_error=0.0;
    // for(size_t iteration=0;iteration<5000000;iteration++)
    // {
    //     for(size_t batch=0;batch<1;batch++)
    //     {
    //         X[0][last_input_index]=0.0;
    //         last_input_index=dst(gen);
    //         X[0][last_input_index]=1.0;

    //         block1->calc(X, 0);
    //         block2->calc(block1->get_output(0), 0);

    //         block2->set_first_delta_and_propagate_with_cross_enthropy(X, block1->get_delta_output(0), 0);
    //         block1->propagate_delta(0);

    //         block1->accumulate_gradients(X,0);
    //         block2->accumulate_gradients(block1->get_output(0),0);
    //     }
    //     block1->update_weights_adam(.001, .999, .9);
    //     block2->update_weights_adam(.001, .999, .9);

    //     if(iteration%100000==0)
    //     {
    //         double error=0.0;
    //         for(size_t new_input_index=0;new_input_index<input_size;new_input_index++)
    //         {
    //             X[0][last_input_index]=0.0;
    //             last_input_index=new_input_index;
    //             X[0][last_input_index]=1.0;

    //             block1->calc(X, 0);
    //             block2->calc(block1->get_output(0), 0);

    //             for(size_t i=0;i<input_size;i++)
    //             {
    //                 double aux=block2->get_output(0)[0][i]-X[0][i];
    //                 aux*=aux;
    //                 error+=aux;
    //             }
    //         }
    //         // error/=input_size;
    //         print(error);
    //     }
    // }
    // print("############");
    // for(size_t i=0;i<input_size;i++)
    // {
    //     X[0][last_input_index]=0.0;
    //     last_input_index=i;
    //     X[0][last_input_index]=1.0;

    //     block1->calc(X, 0);
    //     // block2->calc(block1->get_output(0), 0);
    //     print(block1->get_output(0));
    // }
    // auto first_time=chrono::steady_clock::now()-chrono::seconds{43510};
    // auto elapsed_time=chrono::steady_clock::now()-first_time;
    // string elapsed_time_string(256, '\x00');
    // auto elapsed_time_string_size=sprintf(&elapsed_time_string[0], "%02i:%02i:%02li",
    //     (chrono::duration_cast<chrono::hours>(elapsed_time)).count(),
    //     (chrono::duration_cast<chrono::minutes>(elapsed_time)).count()%60,
    //     (chrono::duration_cast<chrono::seconds>(elapsed_time)).count()%60);
    // elapsed_time_string.resize(elapsed_time_string_size);
    // print(elapsed_time_string);
    assert(0);
    return 0;
}