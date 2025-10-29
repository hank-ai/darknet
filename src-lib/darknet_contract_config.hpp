#pragma once

#include <stdexcept>
#include <string>
#include <sstream>

// Include Boost.Contract headers if needed
#ifdef BOOST_CONTRACT_VERSION
#include <boost/contract.hpp>
#endif

// Custom exception class for contract violations with detailed diagnostics
class contract_violation : public std::runtime_error {
public:
    contract_violation(const std::string& expression, 
                      const std::string& file, 
                      int line,
                      const std::string& function_name = "",
                      const std::string& context = "")
        : std::runtime_error(build_message(expression, file, line, function_name, context))
        , expression_(expression)
        , file_(file)
        , line_(line)
        , function_name_(function_name)
        , context_(context)
    {}

    const std::string& expression() const noexcept { return expression_; }
    const std::string& file() const noexcept { return file_; }
    int line() const noexcept { return line_; }
    const std::string& function_name() const noexcept { return function_name_; }
    const std::string& context() const noexcept { return context_; }

private:
    std::string expression_;
    std::string file_;
    int line_;
    std::string function_name_;
    std::string context_;

    static std::string build_message(const std::string& expression,
                                   const std::string& file,
                                   int line,
                                   const std::string& function_name,
                                   const std::string& context) {
        std::ostringstream msg;
        msg << "Contract violation: \"" << expression << "\" failed";
        msg << " in file \"" << file << "\", line " << line;
        if (!function_name.empty()) {
            msg << " in function \"" << function_name << "\"";
        }
        if (!context.empty()) {
            msg << "\nContext: " << context;
        }
        
        // Add GPU/CPU compilation mode information
        #ifdef GPU
            msg << "\nCompilation mode: GPU enabled";
        #else
            msg << "\nCompilation mode: CPU only";
        #endif
        
        return msg.str();
    }
};

// Custom failure handler for Boost.Contract (only if Boost.Contract is available)
#ifdef BOOST_CONTRACT_VERSION
inline void contract_failure_handler(const boost::contract::from& /* from */) {
    // This handler will be called when a contract fails
    // For now, we'll use the default behavior but this can be customized
    std::terminate();
}

// Configure Boost.Contract to use our custom handler
#define BOOST_CONTRACT_FAILURE_HANDLER contract_failure_handler
#endif

// Helper macro for creating contract violations with context
#define DARKNET_CONTRACT_ASSERT(expr, context) \
    do { \
        if (!(expr)) { \
            throw contract_violation(#expr, __FILE__, __LINE__, __FUNCTION__, context); \
        } \
    } while(0)

// Helper macro for preconditions with context
#define DARKNET_PRECONDITION(expr, context) \
    DARKNET_CONTRACT_ASSERT(expr, "Precondition: " + std::string(context))

// Helper macro for postconditions with context  
#define DARKNET_POSTCONDITION(expr, context) \
    DARKNET_CONTRACT_ASSERT(expr, "Postcondition: " + std::string(context))
