#ifndef __MYSTUFF335503__IAMTONI__
#define __MYSTUFF335503__IAMTONI__
#include <exception>
#include <iostream>
#include <string>

//Exception to be thrown on assertion fails.
class AssertionException : public std::exception
{
private:
    std::string error;
public:
    AssertionException(const char *file, const char *func, int line, const char* assertion)
    {
        error.reserve(512);
        error.append("\n");
        error.append("File:       ");
        error.append(file);
        error.append("\n");
        error.append("Function:   ");
        error.append(func);
        error.append("\n");
        error.append("Line:       ");
        error.append(std::to_string(line));
        error.append("\n");
        error.append("Assertion:  ");
        error.append(assertion);
        error.append("\n");
    }

    AssertionException(const char *message, const char *file, const char *func, int line, const char* assertion)
    {
        error.reserve(1024);
        error.append("\n");
        error.append("Message:    ");
        error.append(message);
        error.append("\n");
        error.append("File:       ");
        error.append(file);
        error.append("\n");
        error.append("Function:   ");
        error.append(func);
        error.append("\n");
        error.append("Line:       ");
        error.append(std::to_string(line));
        error.append("\n");
        error.append("Assertion:  ");
        error.append(assertion);
        error.append("\n");
    }

    virtual const char* what() const noexcept override
    {
        return error.c_str();
    }

    ~AssertionException()noexcept override{}
};

//Asserts "c", throws "AssertionException" on fail.
#define assert(c){\
    if(c){}else throw AssertionException(__FILE__, __func__, __LINE__, #c);\
}

#define assertm(c,m){\
    if(c){}else throw AssertionException(m, __FILE__, __func__, __LINE__, #c);\
}

inline void print()
{
    std::cout << std::endl;
}

// template<typename T>
// inline void print(T &v)
// {
//     std::cout << v << std::endl;
// }

// template<typename T, typename... Args>
// inline void print(T &v, Args... args)
// {
//     std::cout << v << " ";
//     print(args...);
// }

template<typename T>
inline void print(T v)
{
    std::cout << v << std::endl;
}

template<typename T, typename... Args>
inline void print(T v, Args... args)
{
    std::cout << v << " ";
    print(args...);
}


#endif