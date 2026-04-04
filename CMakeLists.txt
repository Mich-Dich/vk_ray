cmake_minimum_required(VERSION 3.25)

project("vk_ray")

find_package(Vulkan 1.3 COMPONENTS dxc)

# ============ DEPENDENCY OPTIONS ============
option(VK_RAY_BUILD_DENOISERS "Build denoisers" OFF)
option(VK_RAY_BUILD_VULKAN_BUILDER "Build bootsraps for easy Vulkan Initialization" ON)

# NEW: Choose between internal or external dependencies
option(VK_RAY_USE_EXTERNAL_DEPS "Use external dependencies (from parent vendor/ directory) instead of internal submodules" ON)

# ============ DEPENDENCIES ============
if(VK_RAY_BUILD_VULKAN_BUILDER)
    if(VK_RAY_USE_EXTERNAL_DEPS)
        # Use external dependencies from parent vendor/ directory
        message(STATUS "vk_ray: Using external dependencies from vendor/")

        # vk-bootstrap is already built in parent vendor/CMakeLists.txt
        # We just need to find the target
        if(NOT TARGET vk-bootstrap)
            message(FATAL_ERROR "External vk-bootstrap target not found. Make sure vendor/CMakeLists.txt builds it first.")
        endif()

        # VMA is included via glm interface target in vendor/CMakeLists.txt
        if(NOT TARGET glm)
            message(FATAL_ERROR "External glm (with VMA) target not found. Make sure vendor/CMakeLists.txt builds it first.")
        endif()
    else()
        # Use internal submodules (original behavior)
        message(STATUS "vk_ray: Using internal submodule dependencies")

        # Add vk-bootstrap subdirectory
        add_subdirectory("${PROJECT_SOURCE_DIR}/vendor/vk-bootstrap/")

        # For VMA, we need to add it manually since it's not in vk_ray's vendor dir
        # You might want to add a check if it exists
        if(EXISTS "${PROJECT_SOURCE_DIR}/vendor/VulkanMemoryAllocator/")
            add_subdirectory("${PROJECT_SOURCE_DIR}/vendor/VulkanMemoryAllocator/")
        else()
            message(WARNING "Internal VMA not found at ${PROJECT_SOURCE_DIR}/vendor/VulkanMemoryAllocator/")
        endif()
    endif()
endif()

# ============ SOURCE FILES ============
# Add all source files for minimal build
file(GLOB VK_RAY_SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")

# Add files depending on options
if(VK_RAY_BUILD_VULKAN_BUILDER)
    list(APPEND VK_RAY_SRC_FILES "${PROJECT_SOURCE_DIR}/src/builders/builders.cpp")
endif()
if(VK_RAY_BUILD_DENOISERS)
    file(GLOB DENOISER_SRC_FILES "${PROJECT_SOURCE_DIR}/src/Denoisers/*.cpp")
    list(APPEND VK_RAY_SRC_FILES ${DENOISER_SRC_FILES})
endif()

add_library("vk_ray" STATIC ${VK_RAY_SRC_FILES})

if(VK_RAY_BUILD_DENOISERS)
    target_compile_definitions("vk_ray" PUBLIC "VK_RAY_BUILD_DENOISERS")
endif()

# ============ HEADER FILE OPTIONS ============
target_precompile_headers("vk_ray" PRIVATE "${PROJECT_SOURCE_DIR}/src/pch.h")

target_include_directories("vk_ray" PUBLIC
    "${Vulkan_INCLUDE_DIRS}"
    "${PROJECT_SOURCE_DIR}/include/"
)

# Handle VMA include path based on dependency choice
if(VK_RAY_BUILD_VULKAN_BUILDER)
    if(VK_RAY_USE_EXTERNAL_DEPS)
        # External VMA is included via glm target from parent vendor/
        target_link_libraries("vk_ray" PRIVATE glm)
    else()
        # Internal VMA path
        target_include_directories("vk_ray" PUBLIC
            "${PROJECT_SOURCE_DIR}/vendor/VulkanMemoryAllocator/include/"
        )
    endif()
endif()

# Handle vk-bootstrap include path based on dependency choice
if(VK_RAY_BUILD_VULKAN_BUILDER)
    if(VK_RAY_USE_EXTERNAL_DEPS)
        # External vk-bootstrap includes are already handled by the target
        target_link_libraries("vk_ray" PRIVATE vk-bootstrap)
    else()
        # Internal vk-bootstrap path
        target_include_directories("vk_ray" PRIVATE "${PROJECT_SOURCE_DIR}/vendor/vk-bootstrap/src/")
        target_link_libraries("vk_ray" PRIVATE "vk-bootstrap")
    endif()
endif()

if(VK_RAY_BUILD_DENOISERS)
    target_include_directories("vk_ray" PRIVATE "${PROJECT_SOURCE_DIR}/Shaders/Bin/") # For denoiser shaders
endif()

# ============ LINK LIBRARIES ============
target_link_libraries("vk_ray" PRIVATE ${Vulkan_LIBRARIES})

set_property(TARGET "vk_ray" PROPERTY CXX_STANDARD 20)

# ============ COMPILE DENOISER SHADERS ============
if(VK_RAY_BUILD_DENOISERS)
    # Create a custom target for compiling the denoiser shaders
    add_custom_target("vk_rayDenoiserShaders")

    file(GLOB DENOISER_SHADER_FILES "${PROJECT_SOURCE_DIR}/Shaders/*.hlsl")

    # Make sure the output directory exists
    file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/Shaders/Bin/")

    foreach(SHADER_FILE ${DENOISER_SHADER_FILES})
        get_filename_component(SHADER_FILENAME "${SHADER_FILE}" NAME_WE)
        set(SHADER_OUTPUT "${PROJECT_SOURCE_DIR}/Shaders/Bin/${SHADER_FILENAME}.spv.h")

        add_custom_command(TARGET "vk_rayDenoiserShaders"
            COMMENT "Compiling shader ${SHADER_FILE}"
            COMMAND ${Vulkan_dxc_EXECUTABLE}
            -T cs_6_5
            -E ${SHADER_FILENAME}_main
            -spirv
            -O3
            -fspv-target-env=vulkan1.3
            -Qstrip_debug
            -Fh "${SHADER_OUTPUT}"
            "${SHADER_FILE}"
        )
    endforeach()

    # Make sure the shaders are compiled before building the library
    add_dependencies("vk_ray" "vk_rayDenoiserShaders")
endif()

# Print configuration summary
message(STATUS "vk_ray Configuration:")
message(STATUS "  - VK_RAY_BUILD_DENOISERS: ${VK_RAY_BUILD_DENOISERS}")
message(STATUS "  - VK_RAY_BUILD_VULKAN_BUILDER: ${VK_RAY_BUILD_VULKAN_BUILDER}")
message(STATUS "  - VK_RAY_USE_EXTERNAL_DEPS: ${VK_RAY_USE_EXTERNAL_DEPS}")
if(VK_RAY_USE_EXTERNAL_DEPS)
    message(STATUS "    Using external vk-bootstrap and VMA from vendor/")
else()
    message(STATUS "    Using internal submodule dependencies")
endif()
