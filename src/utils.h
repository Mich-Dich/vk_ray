#pragma once

#include <iostream>
#include <format>
#include <thread>
#include <functional>

// FORWARD DECLARATIONS ================================================================================================

namespace vr::logging {

    // CONSTANTS =======================================================================================================

    // MACROS ==========================================================================================================

    // This enables the different log levels (FATAL + ERROR are always on)
    //  0 = ERROR
    //  1 = ERROR + WARNING
    //  2 = ERROR + WARNING + INFO
    //  3 = ERROR + WARNING + INFO + VERBOSE
    #define VR_LOG_LEVEL_ENABLED 3              // log all messages by default

    // TYPES ===========================================================================================================

    enum class severity {
        verbose = 0,
        info,
        warning,
        error,
    };

    // STATIC VARIABLES ================================================================================================

    // FUNCTION DECLARATION ============================================================================================

    inline const char* severity_to_string(const severity sev) {
        switch (sev) {
            case severity::verbose:     return "verbose";
            case severity::info:        return "info";
            case severity::warning:     return "warning";
            case severity::error:       return "error";
            default:                    return "unknown severity";
        }
    }

    // TEMPLATE DECLARATION ============================================================================================

    // Template version that uses std::format for format strings with arguments
    template<typename... Args>
    inline void log_msg(const severity msg_sev, const char* file_name, const char* function_name, const int line, std::thread::id thread_id, std::format_string<Args...> fmt, Args&&... args) {

        std::string message = std::format(fmt, std::forward<Args>(args)...);

        uint16_t hour{}, min{}, second{}, millisecond{};
        #if defined(PLATFORM_WINDOWS)
            SYSTEMTIME win_time;
            GetLocalTime(&win_time);
            hour = win_time.wHour;
            min = win_time.wMinute;
            second = win_time.wSecond;
            millisecond = win_time.wMilliseconds;
        #elif defined(PLATFORM_LINUX)
            struct timeval tv;
            gettimeofday(&tv, NULL);
            struct tm* ptm = localtime(&tv.tv_sec);
            hour = ptm->tm_hour;
            min = ptm->tm_min;
            second = ptm->tm_sec;
            millisecond = tv.tv_usec / 1000;
        #endif

        std::cout << std::format("[{:02}:{:02}:{:02}:{:04}] [{} {} {}:{}] {}",
            hour, min, second, millisecond,
            severity_to_string(msg_sev), file_name, function_name, line,
            message);
    }

    // Define a function pointer type for the log_msg function
    using log_function_handle = std::function<void( severity, const char*, const char*, int, std::thread::id, std::string )>;

    // Global function handle for logging - can be replaced by user
    inline log_function_handle current_function_handle = [](severity msg_sev, const char* file_name, const char* function_name, int line, std::thread::id thread_id, std::string message) {

        // Default implementation - use the template function with formatted string
        uint16_t hour{}, min{}, second{}, millisecond{};
        #if defined(PLATFORM_WINDOWS)
            SYSTEMTIME win_time;
            GetLocalTime(&win_time);
            hour = win_time.wHour;
            min = win_time.wMinute;
            second = win_time.wSecond;
            millisecond = win_time.wMilliseconds;
        #elif defined(PLATFORM_LINUX)
            struct timeval tv;
            gettimeofday(&tv, NULL);
            struct tm* ptm = localtime(&tv.tv_sec);
            hour = ptm->tm_hour;
            min = ptm->tm_min;
            second = ptm->tm_sec;
            millisecond = tv.tv_usec / 1000;
        #endif

        std::cout << std::format("[{:02}:{:02}:{:02}:{:04}] [{} {} {}:{}] {}",
            hour, min, second, millisecond,                                             // time info
            severity_to_string(msg_sev), file_name, function_name, line,                // source info
            message);                                                                   // message
    };

    // Helper function to set custom logging function
    inline void set_log_function(log_function_handle new_log_function)      { current_function_handle = std::move(new_log_function); }

    // Template function that uses the function handle
    template<typename... Args>
    inline void log_msg_handle(const severity msg_sev, const char* file_name, const char* function_name, const int line, std::thread::id thread_id, std::format_string<Args...> fmt, Args&&... args) {
        std::string message = std::format(fmt, std::forward<Args>(args)...);
        current_function_handle(msg_sev, file_name, function_name, line, thread_id, std::move(message));
    }

    // MACROS ==========================================================================================================

    #define VR_LOG_master(severity_level, fmt, ...)                                                                                                                            \
        { vr::logging::log_msg_handle(vr::logging::severity::severity_level, __FILE__, __FUNCTION__, __LINE__, std::this_thread::get_id(), fmt __VA_OPT__(,) __VA_ARGS__); }


    #define VR_LOG_error(fmt, ...)              VR_LOG_master(error, fmt __VA_OPT__(,) __VA_ARGS__)

    #if VR_LOG_LEVEL_ENABLED > 0
        #define VR_LOG_warning(fmt, ...)        VR_LOG_master(warning, fmt __VA_OPT__(,) __VA_ARGS__)
    #else
        #define VR_LOG_warning(fmt, ...)        { }
    #endif

    #if VR_LOG_LEVEL_ENABLED > 1
        #define VR_LOG_info(fmt, ...)           VR_LOG_master(info, fmt __VA_OPT__(,) __VA_ARGS__)
    #else
        #define VR_LOG_info(fmt, ...)           { }
    #endif

    #if VR_LOG_LEVEL_ENABLED > 2
        #define VR_LOG_verbose(fmt, ...)        VR_LOG_master(verbose, fmt __VA_OPT__(,) __VA_ARGS__)
    #else
        #define VR_LOG_verbose(fmt, ...)        { }
    #endif

    #define VR_LOG(severity, fmt, ...)          VR_LOG_##severity(fmt __VA_OPT__(,) __VA_ARGS__)

    // CLASS DECLARATION ===============================================================================================

}
