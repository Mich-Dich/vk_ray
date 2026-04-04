
#pragma once



// FORWARD DECLARATIONS ================================================================================================

namespace vr {

    // CONSTANTS =======================================================================================================

    // MACROS ==========================================================================================================

    #if defined(__GNUC__)

        #define IGNORE_UNUSED_VARIABLE_START            _Pragma("GCC diagnostic push")                              \
                                                        _Pragma("GCC diagnostic ignored \"-Wunused-variable\"")

        #define IGNORE_UNUSED_VARIABLE_STOP             _Pragma("GCC diagnostic pop")

    #elif defined(_MSC_VER)

        #define IGNORE_UNUSED_VARIABLE_START            __pragma(warning(push, 0))                                  \
                                                        __pragma(warning(disable: 4189))

        #define IGNORE_UNUSED_VARIABLE_STOP             __pragma(warning(pop))

    #else

        #define IGNORE_UNUSED_VARIABLE_START
        #define IGNORE_UNUSED_VARIABLE_STOP

    #endif

    // TYPES ===========================================================================================================

    // STATIC VARIABLES ================================================================================================

    // FUNCTION DECLARATION ============================================================================================

    // TEMPLATE DECLARATION ============================================================================================

    // CLASS DECLARATION ===============================================================================================

}
